import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace

# Mapping from metadata column names to their types
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
    """Reads a metadata file and returns a list of configurations and metrics.
    Code is a subset of the template code for the project.

    Args:
        path (Path): Path to the metadata file
        space (ConfigurationSpace): Configuration space to validate configs against

    Raises:
        RuntimeError: If no configs are found that are representable in the space

    Returns:
        Tuple[List[Configuration], List[float]]: Returns Configurations in format of space and their corresponding metrics
    """
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
    metrics = []
    for _, row in metadata.iterrows():
        config_dict = row[config_columns].to_dict()  # type: ignore
        try:
            configs.append(Configuration(configuration_space=space, values=config_dict))
            metrics.append(1 - row["cost"])
        except Exception as exception:
            logging.warning(f"Skipping config as not in space:\n%s\n%s", row, exception)

    if len(configs) == 0:
        raise RuntimeError("No configs found that are representable in the space")

    return configs, metrics
