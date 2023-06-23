from __future__ import annotations
from dataclasses import dataclass

import argparse
import json
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    InCondition,
    Categorical,
)
import pandas as pd
import numpy as np
from functools import partial
from typing import Iterator, Literal, Iterable
from pathlib import Path
from dask.distributed import get_worker
from sklearn.model_selection import StratifiedKFold
from smac import Scenario, HyperparameterOptimizationFacade
from smac.runhistory import TrialValue
from smac.initial_design import AbstractInitialDesign
from smac.runhistory.dataclasses import TrialInfo
from torch.utils.data import DataLoader, Subset
from datasets import load_deep_woods
import torch
from cnn import Model
from torchvision.datasets import ImageFolder

import logging

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent.absolute()
METADATA_FILE = HERE / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"
DATA_PATH = HERE / "datasets"

# These should not be changed when doing the final reporting,
# however feel free to change these while experimenting.
MAX_EPOCHS = 20
IMG_SIZE = 32
CV_SPLITS = 3
DEFAULT_RUNTIME = 60 * 60 * 6
DATASET_NAME = "deepweedsx_balanced"

METADATA_CONFIG_COLUMNS = {
    "config:n_conv_layers": int,
    "config:use_BN": bool,
    "config:global_avg_pooling": bool,
    "config:n_channels_conv_0": int,
    "config:n_channels_conv_1": pd.Int64Dtype(),
    "config:n_channels_conv_2": pd.Int64Dtype(),
    "config:n_fc_layers": int,
    "config:n_channels_fc_0": int,
    "config:n_channels_fc_1": pd.Int64Dtype(),
    "config:n_channels_fc_2": pd.Int64Dtype(),
    "config:batch_size": int,
    "config:learning_rate_init": float,
    "config:kernel_size": int,
    "config:dropout_rate": float,
}


class ProvidedInitialDesign(AbstractInitialDesign):
    """Initial design that uses a user-provided list of configurations."""

    def __init__(self, scenario: Scenario, configs: Iterable[Configuration]):
        self.configs = list(configs)
        super().__init__(scenario=scenario, n_configs=len(self.configs))

    def _select_configurations(self) -> list[Configuration]:
        for config in self.configs:
            config.origin = "Provided Initial Design"

        return self.configs


@dataclass
class WarmstartConfig:
    config: Configuration
    seed: int
    cost: float
    duration: float
    budget: float | None = None

    def as_trial(self) -> tuple[TrialInfo, TrialValue]:
        """Converts this WarmstartConfig into a TrialInfo and TrialValue.

        This can be used with `optimizer.tell(info, value)` to inform SMAC about
        a result before the optimization starts.
        """
        # Since we're not using Multi-fidelity, budget=self.budget,
        trial_info = TrialInfo(config=self.config, instance=None, seed=self.seed)
        trial_value = TrialValue(time=self.duration, cost=self.cost)
        return trial_info, trial_value

    @classmethod
    def from_metadata(
        cls,
        path: Path,
        space: ConfigurationSpace,
    ) -> list[WarmstartConfig]:
        metadata = (
            pd.read_csv(path)
            .astype(METADATA_CONFIG_COLUMNS)
            .rename(columns=lambda c: c.replace("config:", ""))
            .drop(
                columns=[
                    "dataset",
                    "datasetpath",
                    "device",
                    "cv_count",
                    "budget_type",
                    "config_id",
                ]
            )
        )

        config_columns = [c.replace("config:", "") for c in METADATA_CONFIG_COLUMNS]

        configs = []
        for _, row in metadata.iterrows():
            config_dict = row[config_columns].to_dict()  # type: ignore
            try:
                configs.append(
                    WarmstartConfig(
                        config=Configuration(
                            configuration_space=space, values=config_dict
                        ),
                        seed=int(row["seed"]),
                        budget=float(row["budget"]),
                        cost=float(row["cost"]),
                        duration=float(row["time"]),
                    )
                )
            except Exception as e:
                logging.warning(f"Skipping config as not in space:\n{row}\n{e}")

        if len(configs) == 0:
            raise RuntimeError("No configs found that are representable in the space")

        return configs


def configuration_space() -> ConfigurationSpace:
    """Build Configuration Space which defines all parameters and their ranges."""
    # This serves only as an example of how you can manually define a Configuration Space
    # To illustrate different parameter types;
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace(
        {
            "n_conv_layers": Integer("n_conv_layers", (1, 3), default=3),
            "use_BN": Categorical("use_BN", [True, False], default=True),
            "global_avg_pooling": Categorical(
                "global_avg_pooling", [True, False], default=True
            ),
            "n_channels_conv_0": Integer(
                "n_channels_conv_0", (32, 512), default=512, log=True
            ),
            "n_channels_conv_1": Integer(
                "n_channels_conv_1", (16, 512), default=512, log=True
            ),
            "n_channels_conv_2": Integer(
                "n_channels_conv_2", (16, 512), default=512, log=True
            ),
            "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3),
            "n_channels_fc_0": Integer(
                "n_channels_fc_0", (32, 512), default=512, log=True
            ),
            "n_channels_fc_1": Integer(
                "n_channels_fc_1", (16, 512), default=512, log=True
            ),
            "n_channels_fc_2": Integer(
                "n_channels_fc_2", (16, 512), default=512, log=True
            ),
            "batch_size": Integer("batch_size", (1, 1000), default=200, log=True),
            "learning_rate_init": Float(
                "learning_rate_init",
                (1e-5, 1.0),
                default=1e-3,
                log=True,
            ),
            "kernel_size": Constant("kernel_size", 3),
            "dropout_rate": Constant("dropout_rate", 0.2),
        }
    )

    # Add multiple conditions on hyperparameters at once:
    cs.add_conditions(
        [
            InCondition(cs["n_channels_conv_2"], cs["n_conv_layers"], [3]),
            InCondition(cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3]),
            InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3]),
            InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3]),
        ]
    )
    return cs


@dataclass
class Data:
    """Encapsulates all data functionality

    Notably:
        * `train_val_splits()`: The splits of the training data for cross-validation
        * `train_test()`: The test data and the data to train on when testing
    """

    train_val: ImageFolder
    test: ImageFolder
    cv: StratifiedKFold
    batch_size: int
    random_state: int
    input_shape: tuple[int, int, int]
    classes: list[str]
    folds: int = CV_SPLITS

    @classmethod
    def from_path(
        cls,
        datapath: Path = DATA_PATH,
        batch_size: int = 32,
        download: bool = True,
        img_size: int = IMG_SIZE,
        folds: int = CV_SPLITS,
        seed: int = 0,
    ) -> Data:
        input_shape, train_val, test = load_deep_woods(
            datadir=datapath / "deepweedsx",
            resize=(img_size, img_size),
            balanced=True,
            download=download,
        )
        cv = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        return Data(
            train_val=train_val,
            test=test,
            cv=cv,
            input_shape=input_shape,
            batch_size=batch_size,
            random_state=seed,
            classes=train_val.classes,
        )

    def train_test(self) -> tuple[DataLoader, DataLoader]:
        train = DataLoader(
            dataset=self.train_val, batch_size=self.batch_size, shuffle=True
        )
        test = DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False)
        return train, test

    def train_val_splits(self) -> Iterator[tuple[DataLoader, DataLoader]]:
        splits = self.cv.split(X=self.train_val, y=self.train_val.targets)
        for train_idx, valid_idx in splits:
            train_loader = DataLoader(
                dataset=Subset(self.train_val, list(train_idx)),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                dataset=Subset(self.train_val, list(valid_idx)),
                batch_size=self.batch_size,
                shuffle=False,
            )
            yield train_loader, val_loader


def test_cnn(
    cfg: Configuration,
    seed: int,
    datapath: Path,
    download: bool = True,
    device: Literal["cpu", "cuda"] = "cpu",
) -> float:
    """Function used to get the test score of a CNN.

    Args:
        cfg: Configuration chosen by smac
        seed: used to initialize the rf's random generator
        datapath: path to the dataset

    Returns:
        Test accuracy
    """
    lr = cfg["learning_rate_init"]
    batch_size = cfg["batch_size"]
    model_optimizer = torch.optim.Adam
    train_criterion = torch.nn.CrossEntropyLoss

    epochs = MAX_EPOCHS
    img_size = IMG_SIZE

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    data = Data.from_path(
        datapath=datapath,
        batch_size=batch_size,
        download=download,
        img_size=img_size,
        seed=seed,
    )

    train_set, test_set = data.train_test()
    logging.info(f"Training on {len(train_set)} batches")

    model = Model(
        config=dict(cfg),
        input_shape=data.input_shape,
        num_classes=len(data.classes),
    ).to(model_device)
    optimizer = model_optimizer(model.parameters(), lr=lr)
    criterion = train_criterion().to(device)

    for epoch in range(epochs):
        logging.info("#" * 50)
        logging.info(f"Epoch [{epoch + 1}/{epochs}]")
        train_score, train_loss = model.train_fn(
            optimizer=optimizer,
            criterion=criterion,
            loader=train_set,
            device=model_device,
        )
        logging.info(f"Train acc. {train_score:.3f} | loss {train_loss}")

    test_score = float(model.eval_fn(test_set, device))
    logging.info(f"Test accuracy {test_score:.3f}")

    return test_score


def cnn_from_cfg(
    cfg: Configuration,
    seed: int,
    datapath: Path,
    download: bool = True,
    device: Literal["cpu", "cuda"] = "cpu",
) -> float:
    """Target function optimized to train a CNN on the dataset

    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    Args:
        cfg: Configuration chosen by smac
        seed: used to initialize the rf's random generator
        datapath: path to the dataset

    Returns:
        cross-validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    lr = cfg["learning_rate_init"]
    batch_size = cfg["batch_size"]
    model_optimizer = torch.optim.Adam
    train_criterion = torch.nn.CrossEntropyLoss

    epochs = MAX_EPOCHS
    img_size = IMG_SIZE

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    data = Data.from_path(
        datapath=datapath,
        batch_size=batch_size,
        download=download,
        img_size=img_size,
        seed=seed,
    )
    input_shape = data.input_shape

    score = []

    train_val_splits = data.train_val_splits()
    for cv_index, (train_loader, val_loader) in enumerate(train_val_splits, start=1):
        logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")

        model = Model(
            config=dict(cfg),
            input_shape=input_shape,
            num_classes=len(data.classes),
        ).to(model_device)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        criterion = train_criterion().to(device)

        for epoch in range(epochs):
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{epochs}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=criterion,
                loader=train_loader,
                device=model_device,
            )
            logging.info(
                f"Worker:{worker_id} => Train acc. {train_score:.3f} | loss {train_loss}"
            )

        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")

        score.append(val_score)

    val_error = float(1 - np.mean(score))  # because minimize
    return float(val_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example using deep weed dataset with naive warmstarting"
    )
    parser.add_argument(
        "--experiment-name",
        default="example",
        type=str,
        help="The unique name of the experiment",
    )
    parser.add_argument(
        "--working-dir",
        default=Path(".").absolute(),
        type=Path,
        help="The base path SMAC will run from",
    )
    parser.add_argument(
        "--runtime",
        default=DEFAULT_RUNTIME,
        type=int,
        help="Max running time (seconds) allocated to run HPO",
    )
    parser.add_argument(
        "--no-test",
        action="store_false",
        help="Whether to evaluate incumbents on the test set after optimization",
    )
    parser.add_argument(
        "--datasetpath",
        type=Path,
        default="data",
        help="Path to directory containing the dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether SMAC should start from scratch and overwrite what's in the experiment dir",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--log-level", default="DEBUG", help="Logging level")
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    experiment_dir = args.working_dir / args.experiment_name

    logging.info(
        f"Running experiment in {experiment_dir} with the following arguments:\n{args=}"
    )

    configspace = configuration_space()
    logging.info(f"Using default space\n {configspace}")

    meta_configs = WarmstartConfig.from_metadata(args.metadata_file, space=configspace)
    logger.info(f"Parsed {len(meta_configs)} meta configs that are in the space")

    scenario = Scenario(
        name=args.experiment_name,
        configspace=configspace,
        deterministic=True,
        output_directory=experiment_dir,
        seed=args.seed,
        n_trials=args.n_trials,
        n_workers=args.n_workers,
        walltime_limit=args.runtime,
    )

    target_function = partial(
        cnn_from_cfg,
        seed=args.seed,
        datapath=args.datasetpath,
        device=args.device,
        download=args.download,
    )
    # See: https://github.com/automl/SMAC3/pull/1045
    target_function.__code__ = cnn_from_cfg.__code__  # type: ignore

    optimizer = HyperparameterOptimizationFacade(
        target_function=target_function,
        scenario=scenario,
        overwrite=args.overwrite,
        logging_level=args.log_level,
        initial_design=None,
    )

    # Start optimization
    incumbent = optimizer.optimize()
    logging.info("Done!")

    results = {
        "args": args.__dict__.copy(),
        "items": [],
    }

    trajectory = optimizer.intensifier.trajectory
    logging.info(trajectory)

    # Record the trajectory
    for item in trajectory:
        config_id = item.config_ids[0]
        config = optimizer.runhistory.get_config(config_id)

        val_cost = item.costs[0]
        assert not isinstance(val_cost, list)

        entry = {
            "val-acc": float(1 - val_cost),
            "walltime": item.walltime,
            "config_id": config_id,
            "config": dict(config),
        }

        if args.evaluate_test:
            try:
                test_accuracy = test_cnn(
                    config,
                    seed=args.seed,
                    datapath=args.datasetpath,
                    device=args.device,
                    download=args.download,
                )
            except Exception as e:
                logging.exception(e)
                test_accuracy = 0.0

            entry["test-acc"] = test_accuracy

        results["items"].append(entry)

    results_path = experiment_dir / "results.json"
    logging.info(f"Writing results to {results_path}")
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=4)

    logging.info(f"Finished writing results to {results_path}")
