import json
import unittest
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_processing.feature_processing.gait_feature_processor import (
    GaitFeatureProcessor,
)


def convert_none_to_nan_and_timestamp(data: Any) -> Any:
    """Recursively convert None values to np.nan and specific entries to pandas Timestamp."""
    if isinstance(data, list):
        return [convert_none_to_nan_and_timestamp(item) for item in data]
    elif isinstance(data, dict):
        # Check for "IC Time" and convert to pandas Timestamp
        if "IC Time" in data:
            data["IC Time"] = [
                pd.to_datetime(t) if t is not None else None for t in data["IC Time"]
            ]
        return {
            key: convert_none_to_nan_and_timestamp(value) for key, value in data.items()
        }
    elif data is None:
        return np.nan
    return data


class TestGaitFeatureProcessor(unittest.TestCase):
    def setUp(self):
        # Load test data from JSON file
        test_data_path = Path("test/data_processing/test_data/test_features.json")
        with open(test_data_path, "r") as f:
            self.test_features = json.load(f)

        # Convert None to np.nan and "IC Time" to pandas Timestamp
        self.test_features = convert_none_to_nan_and_timestamp(self.test_features)

        # Mock IMUData and UserData
        self.mock_imu_data = MagicMock(spec=IMUData)
        self.mock_user_data = MagicMock(spec=UserData)

        # Mock methods and attributes if needed
        self.mock_user_data.user_identifier = "mock_user_id"
        self.mock_imu_data.get_imu_data_id.return_value = "mock_imu_data_id"

        # Initialize the processor
        self.processor = GaitFeatureProcessor()

    def test_process_features(self):
        # Call the method under test
        raw_feature_set_entry, agg_feature_set_entry = self.processor.process_features(
            self.test_features, self.mock_imu_data, self.mock_user_data
        )

        # Assertions to verify the results
        self.assertIsInstance(raw_feature_set_entry, RawFeatureSetEntry)
        self.assertIsInstance(agg_feature_set_entry, AggregateFeatureSetEntry)
        # Assert raw feature types aggregated correctly
        self.assertEqual(len(raw_feature_set_entry.raw_epoch_features), 6)
        self.assertEqual(
            raw_feature_set_entry.metadata.imu_data_identifier,
            self.mock_imu_data.get_imu_data_id(),
        )
        self.assertEqual(
            raw_feature_set_entry.metadata.user_identifier,
            self.mock_user_data.user_identifier,
        )
        for raw_epoch_feature in raw_feature_set_entry.raw_epoch_features:
            self.assertEqual(len(raw_epoch_feature.raw_features), 46)
        # Assert agg feature types computed and aggregated correctly
        self.assertEqual(len(agg_feature_set_entry.aggregate_features), 46)
        self.assertEqual(
            raw_feature_set_entry.metadata.imu_data_identifier,
            self.mock_imu_data.get_imu_data_id(),
        )
        self.assertEqual(
            raw_feature_set_entry.metadata.user_identifier,
            self.mock_user_data.user_identifier,
        )
        for agg_feature in agg_feature_set_entry.aggregate_features:
            self.assertEqual(len(agg_feature.descriptive_statistics), 9)


if __name__ == "__main__":
    unittest.main()
