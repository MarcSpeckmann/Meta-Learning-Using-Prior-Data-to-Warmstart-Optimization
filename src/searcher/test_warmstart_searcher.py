import unittest
from unittest.mock import Mock, patch

from ConfigSpace import (Categorical, Configuration, ConfigurationSpace,
                         Constant, Float, InCondition, Integer)

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
                self.config_space["n_channels_conv_2"], self.config_space["n_conv_layers"], [3]
            ),
            InCondition(
                self.config_space["n_channels_conv_1"], self.config_space["n_conv_layers"], [2, 3]
            ),
            InCondition(
                self.config_space["n_channels_fc_2"], self.config_space["n_fc_layers"], [3]
            ),
            InCondition(
                self.config_space["n_channels_fc_1"], self.config_space["n_fc_layers"], [2, 3]
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

    # Add more test methods to test different aspects of your class
    
    def test_suggest(self):
        # Mocking necessary objects and methods for testing
        mock_trial_id = "trial_1"
        mock_configuration_dict = {"param1": 0.5, "param2": 0.7}
        mock_configuration = Mock()
        mock_configuration_dict_copy = mock_configuration_dict.copy()

        """
        # Configuring the mock objects
        mock_configuration_dict_copy["param1"] = True
        mock_configuration_dict_copy["param2"] = 1
        mock_vars.return_value = mock_configuration_dict_copy
        self.searcher.optimizer.suggest.return_value = mock_configuration_dict
        """

        # Calling the suggest method
        suggested_config = self.searcher.suggest(mock_trial_id)
        print(suggested_config)
        suggested_config = self.searcher.suggest(mock_trial_id)
        print(suggested_config)

        # Assertions
        """
        self.assertEqual(suggested_config, mock_configuration)
        self.assertEqual(self.searcher.configurations[mock_trial_id], mock_configuration_dict)
        self.assertIn(mock_trial_id, self.searcher.running)
        self.searcher.optimizer.suggest.assert_called_once_with(self.searcher.utility)
        mock_vars.assert_called_once_with(mock_configuration)
        """
    
    # ... Add more test methods ...

if __name__ == "__main__":
    unittest.main()
