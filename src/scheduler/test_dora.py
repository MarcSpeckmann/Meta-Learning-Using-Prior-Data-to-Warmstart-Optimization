import unittest
from unittest.mock import MagicMock

from scheduler import Dora


class TestDora(unittest.TestCase):
    def setUp(self):
        self.dora = Dora()  # Create an instance of the Dora class

    def test_prepare_data(self):
        config = {
            "batch_size": 32,
            "dropout_rate": 0.2,
            # ... other configuration options ...
            "val_accuracy_mean": 0.85
        }
        result = {
            "training_iteration": 10,
            "time_total_s": 100.0,
            "val_accuracy_mean": 0.85
        }
        prepared_data = self.dora.prepare_data(config, result)
        self.assertEqual(prepared_data.shape, (1, len(self.dora._hp_names)))
        self.assertEqual(prepared_data.at[0, "batch_size"], 32)
        self.assertEqual(prepared_data.at[0, "dropout_rate"], 0.2)
        # ... assert other values ...

    # Add more test cases for other methods in the Dora class

if __name__ == "__main__":
    unittest.main()
