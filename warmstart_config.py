import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace

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


def config_from_metadata(
    path: Path,
    space: ConfigurationSpace,
) -> Tuple[List[Configuration], List[float]]:
    metadata = (
        pd.read_csv(path)
        .astype(METADATA_CONFIG_COLUMNS)
        .rename(columns=lambda c: c.replace("config:", ""))
        .rename(columns={"use_BN": "use_bn"})
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
    for idx in range(len(config_columns)):
        if config_columns[idx] == "use_BN":
            config_columns[idx] = "use_bn"

    configs = []
    metrics = []
    for _, row in metadata.iterrows():
        config_dict = row[config_columns].to_dict()  # type: ignore
        try:
            configs.append(Configuration(configuration_space=space, values=config_dict))
            metrics.append(1 - row["cost"])
        except Exception as e:
            logging.warning(f"Skipping config as not in space:\n{row}\n{e}")

    if len(configs) == 0:
        raise RuntimeError("No configs found that are representable in the space")

    return configs, metrics
