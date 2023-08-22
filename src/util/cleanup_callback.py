import shutil
from pathlib import Path
from typing import Dict, List

from ray.tune import Callback
from ray.tune.experiment import Trial


class CleanupCallback(Callback):
    """_summary_"""

    def on_trial_complete(
        self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        """Called after a trial instance completed.

        The search algorithm and scheduler are notified before this
        hook is called.

        Arguments:
            iteration (int): Number of iterations of the tuning loop.
            trials (List[Trial]): List of trials.
            trial (Trial): Trial that just has been completed.
            **info: Kwargs dict for forward compatibility.
        """
        print("HeeeOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        print(Path(trial.local_path))
        print(list(Path(trial.local_path).glob("deepweeds_data_*")))

        for folder in Path(trial.local_path).glob("deepweeds_data_*"):
            print(folder)
            shutil.rmtree(folder)

    def on_trial_error(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        """Called after a trial instance failed (errored).

        The search algorithm and scheduler are notified before this
        hook is called.

        Args:
            iteration (int):  Number of iterations of the tuning loop.
            trials (List[Trial]): List of trials.
            trial (Trial): Trial that just has errored.
            **info: Kwargs dict for forward compatibility.
        """
        for folder in Path(trial.local_path).glob("deepweeds_data_*"):
            print(folder)
            shutil.rmtree(folder)
