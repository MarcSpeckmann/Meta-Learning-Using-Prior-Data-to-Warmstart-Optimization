import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Constant,
    Float,
    InCondition,
    Integer,
)
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import FIFOScheduler

from src.model.classification_module import DeepWeedsClassificationModule
from src.model.data_module import DeepWeedsDataModule
from src.searcher.warmstart_searcher import WarmstartSearcher
from src.util.cleanup_callback import CleanupCallback


def objective(config: Configuration) -> None:
    """The objective function for the hyperparameter optimization.

    Args:
        config (Configuration): The configuration of the current trial.
    """
    # Setting the precision to float32 for the matrix multiplication
    # This is needed for the GPU to work properly
    torch.set_float32_matmul_precision("high")
    # Setting a seed for reproducibility.
    pl.seed_everything(SEED, workers=True)

    # Creating our data module. The data module is responsible for loading the data and creating the data loaders.
    # As input we need to provide the configuration of the current trials and some standard configs.
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule
    data_model = DeepWeedsDataModule(
        **config,
        img_size=(IMG_SIZE, IMG_SIZE),
        balanced=BALANCED_DATASET,
        train_val_split=TRAIN_VAL_SPLIT,
        num_workers=DATASET_WORKER_PER_TRIAL,
        data_path=DATA_PATH,
        load_data_on_every_trial=LOAD_DATA_ON_EVERY_TRIAL,
        seed=SEED,
    )

    # Creating our model. The model is responsible for the training and validation of the model.
    # As input we need to provide the configuration of the current trials and some standard configs.
    # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    model = DeepWeedsClassificationModule(
        **config,
        num_classes=NUM_CLASSES,
        input_shape=(3, IMG_SIZE, IMG_SIZE),
        seed=SEED,
    )

    # Creating the callbacks for the training.
    # We need this callbacks to be able to use a (ASHA)Scheduler.
    # The TuneReportCheckpointCallback is responsible for saving the model checkpoints.
    callbacks = [
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.integration.pytorch_lightning.TuneReportCallback.html#ray.tune.integration.pytorch_lightning.TuneReportCallback
        TuneReportCheckpointCallback(
            metrics={OPTIMIZATION_METRIC},
            filename=CHECKPOINT_FILE_NAME,
            on="validation_end",
        ),
    ]
    # The trainer is responsible for the training and validation of the model.
    # As input we need to provide the model, the data module and the callbacks.
    # We also set the max_epochs to the maximum number of epochs we want to train.
    # The deterministic flag is set to True to ensure reproducibility.
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        deterministic=True,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )

    # Starting the training of the model.
    trainer.fit(model, datamodule=data_model)


def main() -> None:
    """Main method of project"""
    # Defining the search space
    # This serves only as an example of how you can manually define a Configuration Space
    # To illustrate different parameter types;
    # we use continuous, integer and categorical parameters.
    config_space = ConfigurationSpace(
        space={
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
        },
        seed=SEED,
    )

    # Add multiple conditions on hyperparameters at once:
    config_space.add_conditions(
        [
            InCondition(
                config_space["n_channels_conv_2"], config_space["n_conv_layers"], [3]
            ),
            InCondition(
                config_space["n_channels_conv_1"], config_space["n_conv_layers"], [2, 3]
            ),
            InCondition(
                config_space["n_channels_fc_2"], config_space["n_fc_layers"], [3]
            ),
            InCondition(
                config_space["n_channels_fc_1"], config_space["n_fc_layers"], [2, 3]
            ),
        ]
    )

    # Defining the tuning settings. The WarmstartSearcher is the part we are implementing for the project.
    # The ASHA scheduler is similar to the HyperBand scheduler, but it prunes trials more aggressive.
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html#ray.tune.schedulers.AsyncHyperBandScheduler
    # The TuneConfig is the main configuration object for the Tune library.
    # Metric is the metric we want to optimize, mode is the direction we want to optimize in. In this case we want to maximize the accuracy.
    # The num_samples is the number of trials we want to run. The time_budget_s is the time limit for the experiment.
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html#ray-tune-tuneconfig
    tune_config = tune.TuneConfig(
        search_alg=WarmstartSearcher(
            config_space=config_space,
            metric=OPTIMIZATION_METRIC,
            mode=OPTIMIZATION_MODE,
            metadata_path=METADATA_FILE,
            seed=SEED,
            max_concurrent=MAX_CONCURRENT_TRIALS,
            add_config_threshold=MAX_EPOCHS,
        ),
        scheduler=FIFOScheduler(),
        metric=OPTIMIZATION_METRIC,
        mode=OPTIMIZATION_MODE,
        num_samples=N_TRIALS,
        time_budget_s=WALLTIME_LIMIT,
        reuse_actors=False,
    )

    # Defining the run configuration. The checkpoint_config is used to save the best models.
    # So we can load them later for testing our best performing model
    # We again define our metric and mode for the checkpointing of the best models.
    # https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html#ray.air.RunConfig
    # https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.CheckpointConfig.html#ray.air.CheckpointConfig
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=KEEP_N_BEST_MODELS,
            checkpoint_score_attribute=OPTIMIZATION_METRIC,
            checkpoint_score_order=OPTIMIZATION_MODE,
        ),
        storage_path=RAY_TUNE_DIR,
        name=EXPERIMENT_NAME,
        callbacks=[CleanupCallback()],
    )

    # Defining the trainable. The trainable is the function that is called for each trial.
    # The tune.with_resources is used to define the resources we want to use for each trial.
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.with_resources.html#ray-tune-with-resources
    trainable = tune.with_resources(
        objective, resources={"CPU": CPU_PER_TRIAL, "GPU": CUDAS_PER_TRIAL}
    )

    # Defining the tuner. The tuner is responsible for the execution of the trials.
    # If we want to resume a previous experiment we can do that by setting the RESUME flag to True.
    # As input we need to provide the trainable, the tune_config, the param_space and the run_config.
    # https://docs.ray.io/en/latest/tune/api/execution.html#tuner
    if not RESUME:
        tuner = tune.Tuner(
            trainable=trainable,
            tune_config=tune_config,
            run_config=run_config,
        )
    else:
        tuner = tune.Tuner.restore(str(RAY_EXPERIMENT_DIR), trainable=trainable)

    # Starting the hyperparameter tuning.
    # Or loading the best hyperparameters if we are not training.
    if TRAIN:
        result_grid = tuner.fit()
    else:
        result_grid = tuner.get_results()
    best_result = result_grid.get_best_result()

    print("\u2500" * os.get_terminal_size().columns)
    print("Best result found were:")
    print("\u2500" * os.get_terminal_size().columns)
    print("Metrics:")
    print(best_result.metrics)
    print("\u2500" * os.get_terminal_size().columns)
    print("Config:")
    print(best_result.config)
    print("\u2500" * os.get_terminal_size().columns)

    # Testing the best model from the hyperparameter tuning.
    if TEST:
        # Getting the checkpoint path to the best model.
        checkpoint_path = best_result.get_best_checkpoint(
            metric=OPTIMIZATION_METRIC, mode=OPTIMIZATION_MODE
        ).path

        # Loading the best model using the checkpoint path.
        t_model = DeepWeedsClassificationModule.load_from_checkpoint(
            checkpoint_path + "/" + CHECKPOINT_FILE_NAME
        )

        # Building the data module with the best hyperparameters.
        d_model = DeepWeedsDataModule(
            **best_result.config,
            img_size=(IMG_SIZE, IMG_SIZE),
            balanced=BALANCED_DATASET,
            train_val_split=TRAIN_VAL_SPLIT,
            num_workers=DATASET_WORKER_PER_TRIAL,
            data_path=DATA_PATH,
            load_data_on_every_trial=False,
            seed=SEED,
        )

        # Testing the best model using the trainer
        t_trainer = pl.Trainer(logger=False, num_nodes=1, devices=1)
        t_trainer.test(model=t_model, datamodule=d_model)


if __name__ == "__main__":
    EXPERIMENT_NAME = "EXPERIMENT_FIFO_WARMSTARTSEARCH_2"  # Name of folder where the experiment is saved
    TRAIN = (
        True  # If True, the experiment is trained, else the best results are loaded.
    )
    TEST = True  # If True, the best model is tested.
    RESUME = False  # If True, the experiment is resumed from a previous checkpoint. Else a new experiment is started.
    SEED = 255462424  # Seed for reproducibility
    N_TRIALS = -1  # Number of trials to run. If -1, the number of trials is infinite.
    WALLTIME_LIMIT = 6 * 60 * 60  # Time limit for the experiment in seconds. 6h
    MAX_EPOCHS = 20  # Maximum number of epochs to train for.
    IMG_SIZE = 32  # Image size to use for the model. (IMG_SIZE, IMG_SIZE)
    MAX_CONCURRENT_TRIALS = 1  # Maximum number of trials to run concurrently.
    DATASET_WORKER_PER_TRIAL = 4  # Number of workers to use for DataLoader.
    CUDAS_PER_TRIAL = 1  # Number of GPUs to use for each trial.
    CPU_PER_TRIAL = 4  # Number of CPUs to use for each trial.
    TRAIN_VAL_SPLIT = 0.2  # Validation split to use for the dataset.
    BALANCED_DATASET = (
        True  # If 1, the dataset is balanced. Else the dataset is not balanced.
    )
    NUM_CLASSES = 8  # Number of classes in the dataset.
    OPTIMIZATION_METRIC = "val_accuracy_mean"  # Metric to optimize for.
    OPTIMIZATION_MODE = "max"  # Mode to optimize for.
    KEEP_N_BEST_MODELS = 1  # Number of best models to keep.
    LOAD_DATA_ON_EVERY_TRIAL = False  # If True, the data is loaded for each trial. Used for distributed training.

    HERE = Path(__file__).parent.absolute()  # Path to this file.
    DATA_PATH = HERE / "data"  # Path to the data directory.
    RAY_TUNE_DIR = HERE / "ray_tune"  # Path to the ray tune directory.
    RAY_EXPERIMENT_DIR = (
        RAY_TUNE_DIR / EXPERIMENT_NAME
    )  # Path to the experiment directory.
    CHECKPOINT_FILE_NAME = "checkpoint.ckpt"  # Name of the checkpoint file.
    METADATA_FILE = (
        HERE / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"
    )  # Path to the metadata file for warmstarting.

    # Setting the seed for reproducibility of ray tune
    pl.seed_everything(seed=SEED, workers=True)
    torch.set_float32_matmul_precision("high")

    # TODO: implement cross validation (How should we use successive halving, when using cv)
    main()
