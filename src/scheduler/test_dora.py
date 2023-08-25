import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from dora import Dora
from ray.tune.experiment import Trial


class TestDora(unittest.TestCase):
    def setUp(self):
        self.dora = Dora()  # Create an instance of the Dora class

    def test_prepare_data(self):
        # Define a mock configuration and result
        mock_config = {
            "batch_size": 32,
            "dropout_rate": 0.2,
            # ... other configuration options ...
        }
        mock_result = {
            "training_iteration": 10,
            "time_total_s": 100.0,
            "val_accuracy_mean": 0.85
        }

        # Call the prepare_data method
        prepared_data = self.dora.prepare_data(mock_config, mock_result)

        # Check the contents of the prepared DataFrame
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(prepared_data.shape[0], 1)
        self.assertEqual(prepared_data.shape[1], len(self.dora._hp_names))
        self.assertEqual(prepared_data.at[0, "batch_size"], 32)
        self.assertEqual(prepared_data.at[0, "dropout_rate"], 0.2)
        self.assertEqual(prepared_data.at[0, "training_iteration"], 10)
        self.assertEqual(prepared_data.at[0, "time_total_s"], 100.0)
        self.assertEqual(prepared_data.at[0, "val_accuracy_mean"], 0.85)

    def test_on_trial_complete_removes_grace_period(self):
        # Create a mock trial and add it to the grace table
        mock_trial = MagicMock(spec=Trial)
        self.dora.grace_table[mock_trial] = 2

        # Call the on_trial_complete method
        self.dora.on_trial_complete(None, mock_trial, {})

        # Check if the trial is removed from the grace table
        self.assertNotIn(mock_trial, self.dora.grace_table)

    def test_on_trial_add_does_not_throw_errors(self):
        # Create a mock trial and trial runner
        mock_trial = MagicMock(spec=Trial)
        mock_trial_runner = MagicMock()

        # Call the on_trial_add method
        try:
            self.dora.on_trial_add(mock_trial_runner, mock_trial)
        except Exception as e:
            self.fail(f"on_trial_add raised an unexpected exception: {e}")

    def test_on_trial_error_does_not_throw_errors(self):
        # Create a mock trial and trial runner
        mock_trial = MagicMock(spec=Trial)
        mock_trial_runner = MagicMock()

        # Call the on_trial_error method
        try:
            self.dora.on_trial_error(mock_trial_runner, mock_trial)
        except Exception as e:
            self.fail(f"on_trial_error raised an unexpected exception: {e}")

    def test_debug_string(self):
        expected_debug_string = "Using Forest scheduling algorithm."
        debug_string = self.dora.debug_string()
        self.assertEqual(debug_string, expected_debug_string)


if __name__ == "__main__":
    unittest.main()
