import math
from typing import Dict, Optional

import pandas as pd
import ray
from filelock import FileLock
from ray.tune.execution import trial_runner
from ray.tune.experiment import Trial
from ray.tune.schedulers import TrialScheduler
from sklearn.ensemble import RandomForestRegressor


class ForestScheduler(TrialScheduler):
    def __init__(
        self,
        time_attr: str = "training_iteration",
        max_t: int = 100,
        seed: int = None,
    ):
        super().__init__()
        self._time_attr = time_attr
        self._max_t = max_t
        self._regressor = RandomForestRegressor(random_state=seed)
        self._just_trained = False
        self._regressor_fitted = False
        self._hp_names = [
            "batch_size",
            "dropout_rate",
            "global_avg_pooling",
            "kernel_size",
            "learning_rate_init",
            "n_channels_conv_0",
            "n_channels_conv_1",
            "n_channels_conv_2",
            "n_channels_fc_0",
            "n_channels_fc_1",
            "n_channels_fc_2",
            "n_conv_layers",
            "n_fc_layers",
            "use_BN",
            "time_this_iter_s",
            "training_iteration",
            "time_total_s",
            "val_accuracy_mean",
        ]
        self.grace_table = {}
        self._trial_improvements = pd.DataFrame(columns=self._hp_names)
        self._fitting_task = None
        self._seed = seed

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        pass

    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        pass

    def on_trial_result(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ) -> str:
        if result[self._time_attr] >= self._max_t:
            return TrialScheduler.STOP
        if result["training_iteration"] % 2 == 1:
            return TrialScheduler.CONTINUE

        if self._fitting_task is not None:
            ready_tasks, _ = ray.wait([self._fitting_task], num_returns=1, timeout=0)
            if self._fitting_task in ready_tasks:
                with FileLock("regressor.lock"):
                    self._regressor = ray.get(self._fitting_task)
                    print("Regressor updated")
                    self._regressor_fitted = True
                    with FileLock("just_trained.lock"):
                        self._just_trained = True
                self._fitting_task = None

        config_df = self.prepare_data(trial.config, result)
        action = TrialScheduler.CONTINUE
        if self._regressor_fitted:
            prediction = None
            with FileLock("regressor.lock"):
                prediction = self._regressor.predict(
                    config_df.drop(["val_accuracy_mean"], axis=1)
                )
            assert prediction is not None
            print("Prediction: ", prediction[0])
            print("Actual: ", config_df["val_accuracy_mean"][0])

            if not math.isclose(
                prediction[0], config_df["val_accuracy_mean"][0], abs_tol=0.01
            ):
                print(
                    "Prediction too far off, continuing trial to learn more about the space"
                )
                if trial in self.grace_table:
                    self.grace_table[trial] += 1
                    print("Grace period extended")
            else:
                if prediction[0] - config_df["val_accuracy_mean"][0] > 0:
                    print("Prediction better than actual")
                    if trial in self.grace_table:
                        self.grace_table[trial] -= 1
                        print("Grace period reduced to ", self.grace_table[trial])
                        if self.grace_table[trial] == 0:
                            print("Grace period over, stopping trial")
                            action = TrialScheduler.STOP
                    else:
                        self.grace_table[trial] = 2
                        print("Grace period started with a value of 2")
                elif prediction[0] - config_df["val_accuracy_mean"][0] < 0:
                    print("Prediction worse than actual")
                    if trial in self.grace_table:
                        self.grace_table[trial] += 1
                        print("Grace period extended to ", self.grace_table[trial])
                else:
                    print("Prediction equal to actual")

                future_config_df = config_df.copy()
                future_config_df.at[0, "training_iteration"] = (
                    future_config_df.at[0, "training_iteration"] + 2
                )
                future_prediction = None
                with FileLock("regressor.lock"):
                    future_prediction = self._regressor.predict(
                        future_config_df.drop(["val_accuracy_mean"], axis=1)
                    )
                assert future_prediction is not None
                print("Future Prediction: ", prediction[0])
                if not math.isclose(future_prediction[0], prediction[0], abs_tol=0.001):
                    if future_prediction[0] >= config_df["val_accuracy_mean"][0]:
                        print("Predicting config to improve or not worsen, continuing")
                        if trial in self.grace_table:
                            self.grace_table[trial] += 1
                            print("Grace period extended")
                    else:
                        print("Predicting config to worsen, reducing grace period")
                        if trial in self.grace_table:
                            self.grace_table[trial] -= 1
                            if self.grace_table[trial] == 0:
                                print("Grace period over, stopping trial")
                                action = TrialScheduler.STOP
                        else:
                            self.grace_table[trial] = 2
                            print("Grace period started")

        # update performance metrics and update regressor
        with FileLock("trial_improvements.lock"):
            self._trial_improvements = pd.concat(
                [
                    self._trial_improvements,
                    config_df,
                ],
                ignore_index=True,
            )
            print(
                self._trial_improvements.size / len(self._trial_improvements.columns),
                " trial information ready for regressor",
            )
            with FileLock("just_trained.lock"):
                self._just_trained = False
            if self._just_trained is False and self._fitting_task is None:
                self._fitting_task = self.train_regressor.remote(
                    self, self._trial_improvements.copy()
                )
        return action

    def on_trial_complete(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ):
        pass

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        pass

    def choose_trial_to_run(
        self, trial_runner: "trial_runner.TrialRunner"
    ) -> Optional[Trial]:
        for trial in trial_runner.get_trials():
            if (
                trial.status == Trial.PENDING
                and trial_runner.trial_executor.has_resources_for_trial(trial)
            ):
                return trial
        for trial in trial_runner.get_trials():
            if (
                trial.status == Trial.PAUSED
                and trial_runner.trial_executor.has_resources_for_trial(trial)
            ):
                return trial
        return None

    def debug_string(self) -> str:
        return "Using Forest scheduling algorithm."

    @ray.remote
    def train_regressor(self, data: pd.DataFrame):
        print("Training regressor")
        regressor = RandomForestRegressor(random_state=self._seed)
        print(regressor.__dict__)
        return regressor.fit(
            data.drop(columns=["val_accuracy_mean"]),
            data["val_accuracy_mean"],
        )

    def prepare_data(self, config: dict, result: dict):
        config_copy = config.copy()
        config_copy["time_this_iter_s"] = result["time_this_iter_s"]
        config_copy["training_iteration"] = result["training_iteration"]
        config_copy["time_total_s"] = result["time_total_s"]
        config_copy["val_accuracy_mean"] = result["val_accuracy_mean"]
        config_df = pd.DataFrame([config_copy], columns=self._hp_names)
        config_df = config_df.fillna(-1)
        return config_df
