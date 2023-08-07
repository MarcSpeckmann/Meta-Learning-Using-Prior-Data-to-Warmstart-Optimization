from typing import Dict, List, Optional

from ConfigSpace import ConfigurationSpace
from ray import cloudpickle
from ray.tune.search import UNDEFINED_METRIC_MODE, UNDEFINED_SEARCH_SPACE, Searcher

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
        max_concurrent: int = 0,
    ) -> None:
        super().__init__(metric=metric, mode=mode)
        self.search_space = config_space
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
        configuration = self.search_space.sample_configuration()

        # Append the sampled configuration to the list of configurations.
        self.configurations[trial_id] = configuration
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
