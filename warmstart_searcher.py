from pathlib import Path
from typing import Dict, List, Optional

from bayes_opt import BayesianOptimization, UtilityFunction
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Constant,
    Float,
    InCondition,
    Integer,
)
from ray import cloudpickle
from ray.tune.search import UNDEFINED_METRIC_MODE, UNDEFINED_SEARCH_SPACE, Searcher
from scipy.optimize import NonlinearConstraint

from warmstart_config import config_from_metadata


class WarmstartSearcher(Searcher):
    """
    Searcher that uses a warmstart configuration.
    It is a derived class of Searcher, which is a base class for
    writting custom search algorithms for Tune.
    https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.Searcher.html#ray.tune.search.Searcher
    """

    def __init__(
        self,
        metadata_path: str,
        config_space: ConfigurationSpace,
        metric: str,
        mode: str,
        seed: int,
        max_concurrent: int = 0,
    ) -> None:
        super().__init__(metric=metric, mode=mode)
        self.search_space = config_space
        self.seed = seed
        self.warmstart_configs = []
        self.configurations = {}
        self.results = {}
        self._max_concurrent = max_concurrent
        self.running = set()

        # Load the metadata file and convert it to a list of configurations
        self.warmstart_configs, self.warmstart_results = config_from_metadata(
            metadata_path, config_space
        )

        # TODO: use the warmstart_configs and warmstart_results to shrink/extend provided search space
        # TODO: auto generate bounds
        bounds = {
            "batch_size": (1, 1000),
            "global_avg_pooling": (0, 1),
            "learning_rate_init": (1e-5, 1.0),
            "n_channels_conv_0": (32, 512),
            "n_channels_fc_0": (32, 512),
            "n_conv_layers": (1, 3),
            "n_fc_layers": (1, 3),
            "use_BN": (0, 1),
            "n_channels_conv_1": (16, 512),
            "n_channels_conv_2": (16, 512),
            "n_channels_fc_1": (16, 512),
            "n_channels_fc_2": (16, 512),
        }

        self.optimizer = BayesianOptimization(f=None, pbounds=bounds, random_state=seed)
        self.utility = UtilityFunction(kind="ei")

        for config, metric in zip(self.warmstart_configs, self.warmstart_results):
            param = {}
            for key in bounds.keys():
                if key in config.keys():
                    param[key] = config[key]
                else:
                    param[key] = 0
            self.optimizer.register(
                params=param,
                target=metric,
            )

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """Queries the algorithm to retrieve the next set of parameters.

        Arguments:
            trial_id: Trial ID used for subsequent notifications.

        Returns:
            dict | FINISHED | None: Configuration for a trial, if possible.
                If FINISHED is returned, Tune will be notified that
                no more suggestions/configurations will be provided.
                If None is returned, Tune will skip the querying of the
                searcher for this step.

        """
        # Check if the search space is defined.
        if not self.search_space:
            raise RuntimeError(
                UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space="space"
                )
            )

        # Check if the metric and mode are defined.
        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__, metric=self._metric, mode=self._mode
                )
            )

        # Check if is alowed to suggest a new configuration.
        # Its allowed if the number of running trials is less than the max_concurrent
        max_concurrent = (
            self._max_concurrent if self._max_concurrent > 0 else float("inf")
        )

        if len(self.running) >= max_concurrent:
            return None

        # Sample a configuration from the search space
        # TODO: Implement a custom sampling method
        # Important: Pay attention to seeding, when using randomness
        # Otherwise, reproducibility is not guaranteed
        # configuration_dict = self.search_space.sample_configuration()
        configuration_dict = self.optimizer.suggest(self.utility)
        self.configurations[trial_id] = configuration_dict.copy()

        for key, value in configuration_dict.items():
            if key in ["learning_rate_init"]:
                configuration_dict[key] = value
            elif key in ["use_BN", "global_avg_pooling"]:
                configuration_dict[key] = bool(round(value))
            else:
                configuration_dict[key] = round(value)

        configuration_dict["dropout_rate"] = 0.2
        configuration_dict["kernel_size"] = 3

        if configuration_dict["n_conv_layers"] == 1:
            del configuration_dict["n_channels_conv_2"]
            del configuration_dict["n_channels_conv_1"]
        elif configuration_dict["n_conv_layers"] == 2:
            del configuration_dict["n_channels_conv_2"]

        if configuration_dict["n_fc_layers"] == 2:
            del configuration_dict["n_channels_fc_2"]
        elif configuration_dict["n_fc_layers"] == 1:
            del configuration_dict["n_channels_fc_2"]
            del configuration_dict["n_channels_fc_1"]

        configuration = Configuration(self.search_space, configuration_dict)

        # Append the sampled configuration to the list of configurations.
        # self.configurations[trial_id] = configuration_dict
        self.running.add(trial_id)

        return configuration

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ) -> None:
        """Notification for the completion of trial.

        Typically, this method is used for notifying the underlying
        optimizer of the result.

        Args:
            trial_id: A unique string ID for the trial.
            result: Dictionary of metrics for current training progress.
                Note that the result dict may include NaNs or
                may not include the optimization metric. It is up to the
                subclass implementation to preprocess the result to
                avoid breaking the optimization process. Upon errors, this
                may also be None.
            error: True if the training process raised an error.

        """
        self.running.discard(trial_id)
        self.results[trial_id] = result

        self.optimizer.register(
            params=self.configurations[trial_id],
            target=result[self.metric],
        )

    def save(self, checkpoint_path: str) -> None:
        """Save state to path for this search algorithm.

        Args:
            checkpoint_path: File where the search algorithm
                state is saved. This path should be used later when
                restoring from file.

        """
        save_object = self.__dict__
        with open(checkpoint_path, "wb") as output_file:
            cloudpickle.dump(save_object, output_file)

    def restore(self, checkpoint_path: str) -> None:
        """Restore state for this search algorithm


        Args:
            checkpoint_path: File where the search algorithm
                state is saved. This path should be the same
                as the one provided to "save".
        """
        with open(checkpoint_path, "rb") as input_file:
            save_object = cloudpickle.load(input_file)
        self.__dict__.update(save_object)

    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], config: Dict, **spec
    ) -> bool:
        """Pass search properties to searcher.

        This method acts as an alternative to instantiating search algorithms
        with their own specific search spaces. Instead they can accept a
        Tune config through this method. A searcher should return ``True``
        if setting the config was successful, or ``False`` if it was
        unsuccessful, e.g. when the search space has already been set.

        Args:
            metric: Metric to optimize
            mode: One of ["min", "max"]. Direction to optimize.
            config: Tune config dict.
            **spec: Any kwargs for forward compatiblity.
                Info like Experiment.PUBLIC_KEYS is provided through here.
        """
        if self.search_space:
            return False

        self.search_space = config
        if mode:
            self._mode = mode
        if metric:
            self._metric = metric
        return True

    def set_max_concurrency(self, max_concurrent: int) -> bool:
        """Set max concurrent trials this searcher can run.

        This method will be called on the wrapped searcher by the
        ``ConcurrencyLimiter``. It is intended to allow for searchers
        which have custom, internal logic handling max concurrent trials
        to inherit the value passed to ``ConcurrencyLimiter``.

        If this method returns False, it signifies that no special
        logic for handling this case is present in the searcher.

        Args:
            max_concurrent: Number of maximum concurrent trials.
        """
        self._max_concurrent = max_concurrent
        return True

    # Optional section #

    def add_evaluated_point(
        self,
        parameters: Dict,
        value: float,
        error: bool = False,
        pruned: bool = False,
        intermediate_values: Optional[List[float]] = None,
    ):
        """Pass results from a point that has been evaluated separately.

        This method allows for information from outside the
        suggest - on_trial_complete loop to be passed to the search
        algorithm.
        This functionality depends on the underlying search algorithm
        and may not be always available.

        Args:
            parameters: Parameters used for the trial.
            value: Metric value obtained in the trial.
            error: True if the training process raised an error.
            pruned: True if trial was pruned.
            intermediate_values: List of metric values for
                intermediate iterations of the result. None if not
                applicable.

        """
        raise NotImplementedError

    def get_state(self) -> Dict:
        raise NotImplementedError

    def set_state(self, state: Dict):
        raise NotImplementedError

    def on_trial_result(self, trial_id: str, result: Dict) -> None:
        """Optional notification for result during training.

        Note that by default, the result dict may include NaNs or
        may not include the optimization metric. It is up to the
        subclass implementation to preprocess the result to
        avoid breaking the optimization process.

        Args:
            trial_id: A unique string ID for the trial.
            result: Dictionary of metrics for current training progress.
                Note that the result dict may include NaNs or
                may not include the optimization metric. It is up to the
                subclass implementation to preprocess the result to
                avoid breaking the optimization process.
        """
        pass


if __name__ == "__main__":
    SEED = 42
    OPTIMIZATION_METRIC = "val_accuracy_mean"  # Metric to optimize for.
    OPTIMIZATION_MODE = "max"  # Mode to optimize for.
    HERE = Path(__file__).parent.absolute()
    METADATA_FILE = (
        HERE / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"
    )  # Path to the metadata file for warmstarting.

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

    searcher = WarmstartSearcher(
        config_space=config_space,
        metric=OPTIMIZATION_METRIC,
        mode=OPTIMIZATION_MODE,
        metadata_path=METADATA_FILE,
        seed=SEED,
    )

    for i in range(50):
        print(searcher.suggest(i))
        searcher.on_trial_complete(i, {OPTIMIZATION_METRIC: i / 10})