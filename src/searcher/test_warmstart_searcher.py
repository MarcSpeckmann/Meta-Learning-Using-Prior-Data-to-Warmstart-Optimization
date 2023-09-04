import unittest

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    Float,
    InCondition,
    Integer,
)

from src.searcher.warmstart_searcher import WarmstartSearcher


class TestWarmstartSearcher(unittest.TestCase):
    def setUp(self):
        self.metadata_path = "metadata/deepweedsx_balanced-epochs-trimmed.csv"
        self.config_space = ConfigurationSpace(
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
            seed=42,
        )
        self.config_space.add_conditions(
            [
                InCondition(
                    self.config_space["n_channels_conv_2"],
                    self.config_space["n_conv_layers"],
                    [3],
                ),
                InCondition(
                    self.config_space["n_channels_conv_1"],
                    self.config_space["n_conv_layers"],
                    [2, 3],
                ),
                InCondition(
                    self.config_space["n_channels_fc_2"],
                    self.config_space["n_fc_layers"],
                    [3],
                ),
                InCondition(
                    self.config_space["n_channels_fc_1"],
                    self.config_space["n_fc_layers"],
                    [2, 3],
                ),
            ]
        )
        self.metric = "val_accuracy_mean"
        self.mode = "max"
        self.seed = 42
        self.searcher = WarmstartSearcher(
            metadata_path=self.metadata_path,
            config_space=self.config_space,
            metric=self.metric,
            mode=self.mode,
            seed=self.seed,
        )

    def test_suggest(self):
        # Call suggest method multiple times
        config_list = []
        for i in range(1, 51):
            string = "test"
            string += str(i)
            config_list.append(self.searcher.suggest(string))

        self.assertEqual(len(config_list), 50)
        trail = None
        for x in config_list:
            if trail is None:
                trail = x
                continue
            self.assertNotEqual(trail, x)
            trail = x

    def test_on_trial_complete(self):
        trial_id = "testX"
        self.searcher.suggest(trial_id)
        result1 = {"training_iteration": 5, "val_accuracy_mean": 0.5}
        result2 = {"training_iteration": 6, "val_accuracy_mean": 0.6}
        self.searcher.on_trial_result(trial_id, result1)
        self.searcher.on_trial_complete(trial_id, result2)
        try:
            self.searcher.running.remove("testY")
            self.assertEqual(True, False)
        except KeyError:
            self.assertEqual(True, True)
        self.assertEqual(result2 in self.searcher.results[trial_id], True)

        max_metric = max(
            [
                trial_result[self.searcher.metric]
                for trial_result in self.searcher.results[trial_id]
            ]
        )
        x = self.searcher.optimizer._space.params_to_array(
            self.searcher.configurations[trial_id]
        )
        y = self.searcher.optimizer._space._params
        self.assertEqual(x in y, True)
        self.assertEqual(max_metric in self.searcher.optimizer._space._target, True)


if __name__ == "__main__":
    unittest.main()
