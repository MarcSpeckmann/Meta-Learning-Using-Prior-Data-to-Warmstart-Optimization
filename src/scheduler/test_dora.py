import random
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from dora import Dora
from filelock import FileLock
from ray.tune.execution import trial_runner
from ray.tune.experiment import Trial


class TestDora(unittest.TestCase):
    def setUp(self):
        self.dora = Dora()  # Create an instance of the Dora class
        self.dora._test = True

    
    def test_prepare_data(self):
        # Define a mock configuration and result
        mock_config = {
            "batch_size": 32,
            "dropout_rate": 0.2,
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

    def test_prepare_data_missing_values(self):
        # Define a mock configuration and result
        mock_config = {
            "batch_size": None,
            "dropout_rate": None,
        }
        mock_result = {
            "training_iteration": 20,
            "time_total_s": 120.00,
            "val_accuracy_mean": 0.8
        }

        # Call the prepare_data method
        prepared_data = self.dora.prepare_data(mock_config, mock_result)

        # Check the contents of the prepared DataFrame
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(prepared_data.shape[0], 1)
        self.assertEqual(prepared_data.shape[1], len(self.dora._hp_names))
        self.assertEqual(prepared_data.at[0, "batch_size"], -1)
        self.assertEqual(prepared_data.at[0, "dropout_rate"], -1)
        self.assertEqual(prepared_data.at[0, "training_iteration"], 20)
        self.assertEqual(prepared_data.at[0, "time_total_s"], 120.00)
        self.assertEqual(prepared_data.at[0, "val_accuracy_mean"], 0.8)

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


    def test_on_trial_result(self):
        #Create trial runner, and result
        mock_trial_runner = MagicMock()
        mock_result = {
            "training_iteration": 10,
            "time_total_s": 100.0,
            "val_accuracy_mean": 0.85
        }

        # Mock behavior for self._regressor_fitted and self.grace_table
        self.dora._regressor_fitted = True

        mock_regressor = MagicMock()
        def predict(any):
            return mock_regressor.return_value
        mock_regressor.predict = predict
        mock_regressor.return_value = [0.851]  # Mock a better prediction
        self.dora._regressor = mock_regressor
        

        # Call the on_trial_result method
        trial = Trial("Trial", stub=True)
        self.dora.grace_table[trial] = 1
        action = self.dora.on_trial_result(mock_trial_runner, trial, mock_result)

        # Check that grace table is increased and bounded by 2
        self.assertEqual(self.dora.grace_table[trial], 2)

        # Check that action is CONTINUE due to grace period extension
        self.assertEqual(action, self.dora.CONTINUE)

        # Reset grace table for next test
        self.dora.grace_table[trial] = 1

        # Mock worse prediction
        mock_regressor.return_value = [0.849]

        # Call the on_trial_result method
        action = self.dora.on_trial_result(mock_trial_runner, trial, mock_result)

        # Check that grace table is decreased
        self.assertEqual(self.dora.grace_table[trial], 0)

        # Check that action is STOP due to grace table being zero
        self.assertEqual(action, self.dora.STOP)

        # Reset grace table for next test
        self.dora.grace_table[trial] = 1

        # Mock better prediction
        mock_regressor.return_value = [0.852]

        # Call the on_trial_result method
        action = self.dora.on_trial_result(mock_trial_runner, trial, mock_result)

        # Check that grace table is increased and bounded by 2
        self.assertEqual(self.dora.grace_table[trial], 2)

        # Check that action is CONTINUE due to better prediction
        self.assertEqual(action, self.dora.CONTINUE)



if __name__ == "__main__":
    unittest.main()
