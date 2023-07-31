import gc
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

from classification_module import DeepWeedsClassificationModule
from data_module import DeepWeedsDataModule
from warmstart_searcher import WarmstartSearcher


def objective(config):
    """_summary_

    Args:
        config (_type_): _description_
    """
    logging.info("Preparing for training")
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")

    logging.info("Loading dataset")
    data_model = DeepWeedsDataModule(
        batch_size=config["batch_size"],
        img_size=(IMG_SIZE, IMG_SIZE),
        balanced=BALANCED_DATASET,
        train_val_split=TRAIN_VAL_SPLIT,
        num_workers=DATASET_WORKER_PER_TRIAL,
        seed=config["seed"],
    )

    logging.info("Loading model")
    model = DeepWeedsClassificationModule(
        n_conv_layers=config["n_conv_layers"],
        use_bn=config["use_bn"],
        global_avg_pooling=config["global_avg_pooling"],
        n_channels_conv_0=config["n_channels_conv_0"],
        n_channels_conv_1=config["n_channels_conv_1"],
        n_channels_conv_2=config["n_channels_conv_2"],
        n_fc_layers=config["n_fc_layers"],
        n_channels_fc_0=config["n_channels_fc_0"],
        n_channels_fc_1=config["n_channels_fc_1"],
        n_channels_fc_2=config["n_channels_fc_2"],
        learning_rate_init=config["learning_rate_init"],
        kernel_size=config["kernel_size"],
        dropout_rate=config["dropout_rate"],
        num_classes=NUM_CLASSES,
        input_shape=(3, IMG_SIZE, IMG_SIZE),
        seed=config["seed"],
    )

    logging.info("Create callbacks")
    callbacks = [
        TuneReportCallback(
            [
                OPTIMIZATION_METRIC,
            ],
            on="validation_end",
        ),
        TuneReportCheckpointCallback(
            metrics={OPTIMIZATION_METRIC},
            filename=CHECKPOINT_FILE_NAME,
            on="validation_end",
        ),
    ]

    logging.info("Create trainer")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=TensorBoardLogger(LOGS_PATH),
        callbacks=callbacks,
        deterministic=True,
    )

    logging.info("Start training")
    trainer.fit(model, datamodule=data_model)


def main():
    config_space = {
        "n_conv_layers": tune.randint(1, 4),
        "use_bn": tune.randint(0, 2),
        "global_avg_pooling": tune.randint(0, 2),
        "n_channels_conv_0": tune.randint(32, 513),
        "n_channels_conv_1": tune.randint(16, 513),
        "n_channels_conv_2": tune.randint(16, 513),
        "n_fc_layers": tune.randint(1, 4),
        "n_channels_fc_0": tune.randint(32, 513),
        "n_channels_fc_1": tune.randint(16, 513),
        "n_channels_fc_2": tune.randint(16, 513),
        "batch_size": tune.randint(1, 128),
        "learning_rate_init": tune.uniform(1e-5, 1),
        "kernel_size": tune.choice([3]),
        "dropout_rate": tune.choice([0.2]),
        "seed": tune.choice([SEED]),
    }

    tune_config = tune.TuneConfig(
        search_alg=WarmstartSearcher(metadata_path=METADATA_FILE),
        metric=OPTIMIZATION_METRIC,
        mode=OPTIMIZATION_MODE,
        num_samples=N_TRIALS,
        time_budget_s=WALLTIME_LIMIT,
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=KEEP_N_BEST_MODELS,
            checkpoint_score_attribute=OPTIMIZATION_METRIC,
            checkpoint_score_order=OPTIMIZATION_MODE,
        ),
        storage_path=RAY_TUNE_DIR,
        name=EXPERIMENT_NAME,
    )

    trainable = tune.with_resources(
        objective, resources={"CPU": CPU_PER_TRIAL, "GPU": CUDAS_PER_TRIAL}
    )

    if not RESUME:
        tuner = tune.Tuner(
            trainable=trainable,
            tune_config=tune_config,
            param_space=config_space,
            run_config=run_config,
        )
    else:
        tuner = tune.Tuner.restore(str(RAY_EXPERIMENT_DIR), trainable=trainable)

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result()

    print("Best hyperparameters found were: ", best_result.config)

    checkpoint_path = best_result.best_checkpoints[0][0].path

    t_model = DeepWeedsClassificationModule.load_from_checkpoint(
        checkpoint_path + "/" + CHECKPOINT_FILE_NAME
    )

    d_model = DeepWeedsDataModule(
        batch_size=best_result.config["batch_size"],
        img_size=(IMG_SIZE, IMG_SIZE),
        balanced=BALANCED_DATASET,
        train_val_split=TRAIN_VAL_SPLIT,
        num_workers=DATASET_WORKER_PER_TRIAL,
        seed=best_result.config["seed"],
    )

    t_trainer = pl.Trainer()
    t_trainer.test(model=t_model, datamodule=d_model)


if __name__ == "__main__":
    PROJECT_NAME = "Meta-Learning-Using-Prior-Data-to-Warmstart-Optimization"
    EXPERIMENT_NAME = "test_exp"
    RESUME = True
    DATASET_NAME = "deepweedsx_balanced"
    SEED = 42
    N_TRIALS = 1
    WALLTIME_LIMIT = 6 * 60 * 60
    MAX_EPOCHS = 1
    IMG_SIZE = 32
    CV_SPLITS = 3
    DATASET_WORKER_PER_TRIAL = 4
    CUDAS_PER_TRIAL = 0
    CPU_PER_TRIAL = 4
    TRAIN_VAL_SPLIT = 0.1
    BALANCED_DATASET = 1
    NUM_CLASSES = 8
    OPTIMIZATION_METRIC = "val_accuracy_mean"
    OPTIMIZATION_MODE = "max"
    KEEP_N_BEST_MODELS = 2

    HERE = Path(__file__).parent.absolute()
    DATA_PATH = HERE / "data"
    LOGS_PATH = HERE / "tb_logs"
    RAY_TUNE_DIR = HERE / "ray_tune"
    RAY_EXPERIMENT_DIR = RAY_TUNE_DIR / EXPERIMENT_NAME
    CHECKPOINT_FILE_NAME = "checkpoint"
    METADATA_FILE = HERE / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"

    pl.seed_everything(seed=SEED, workers=True)
    torch.set_float32_matmul_precision("high")

    # TODO: implement cross validation (How should we use successive halving, when using cv)
    main()
