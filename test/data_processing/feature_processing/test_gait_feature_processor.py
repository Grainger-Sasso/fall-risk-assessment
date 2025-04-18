import json
import os
import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.data_io.import_export.exporters.feature.aggregate.aggregate_feature_file_exporter import (
    AggregateFeatureFileExporter,
)
from src.data_io.import_export.exporters.feature.raw.raw_feature_file_exporter import (
    RawFeatureFileExporter,
)
from src.data_io.import_export.importers.features.aggregate.aggregate_feature_importer import (
    AggregateFeatureFileNames,
    AggregateFeatureImporter,
)
from src.data_io.import_export.importers.features.raw.raw_feature_importer import (
    RawFeatureFileNames,
    RawFeatureImporter,
)
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_processing.feature_processing.gait_feature_processor import (
    GaitFeatureProcessor,
)
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


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
        self.raw_feature_ID_value = "raw_test_id_value"
        self.agg_feature_ID_value = "agg_test_id_value"

        # Mock IMUData and UserData
        self.mock_imu_data = MagicMock(spec=IMUData)
        self.mock_user_data = MagicMock(spec=UserData)

        # Mock methods and attributes if needed
        self.mock_user_data.user_identifier = UserIdentifier("mock_user_id")
        # self.mock_imu_data.meta = IMUDataIdentifier("mock_imu_data_id")

        # Initialize the processor
        self.processor = GaitFeatureProcessor()

        # Initialize I/O classes
        self.raw_feature_importer = RawFeatureImporter()
        self.raw_feature_exporter = RawFeatureFileExporter()
        self.agg_feature_importer = AggregateFeatureImporter()
        self.agg_feature_exporter = AggregateFeatureFileExporter()

        # Create temp dir for I/O
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

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
            self.mock_imu_data.get_data_id(),
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
            self.mock_imu_data.get_data_id(),
        )
        self.assertEqual(
            raw_feature_set_entry.metadata.user_identifier,
            self.mock_user_data.user_identifier,
        )
        for agg_feature in agg_feature_set_entry.aggregate_features:
            self.assertEqual(len(agg_feature.descriptive_statistics), 9)

    def test_process_features_with_IO(self):
        raw_feature_set_entry, agg_feature_set_entry = self.processor.process_features(
            self.test_features, self.mock_imu_data, self.mock_user_data
        )
        raw_feature_set_entry.metadata.raw_feature_identifier = RawFeatureIdentifier(
            self.raw_feature_ID_value
        )
        raw_feature_set_entry.metadata.imu_data_identifier = IMUDataIdentifier(
            "mock_imu_data_id"
        )
        raw_feature_set_entry.metadata.user_identifier = UserIdentifier("mock_user_id")
        agg_feature_set_entry.metadata.raw_feature_identifier = RawFeatureIdentifier(
            self.raw_feature_ID_value
        )
        agg_feature_set_entry.metadata.aggregate_feature_identifier = (
            AggregateFeatureIdentifier(self.agg_feature_ID_value)
        )
        agg_feature_set_entry.metadata.imu_data_identifier = IMUDataIdentifier(
            "mock_imu_data_id"
        )
        agg_feature_set_entry.metadata.user_identifier = UserIdentifier("mock_user_id")

        # Export the raw features
        raw_output_subdir_path: Path = self.raw_feature_exporter.export_data(
            self.temp_path, raw_feature_set_entry
        )
        # Verify export succeeded with correct output path
        self.assertIsInstance(raw_output_subdir_path, Path)
        # Verify subdirectory was created with correct name
        raw_expected_subdir = (
            self.temp_path / f"raw_features_{self.raw_feature_ID_value}"
        )
        self.assertEqual(str(raw_output_subdir_path), str(raw_expected_subdir))
        self.assertTrue(raw_output_subdir_path.exists(), "Subdirectory not created")
        self.assertTrue(
            raw_output_subdir_path.is_dir(), "Subdirectory is not a directory"
        )
        # Verify file was created with correct name
        raw_expected_file = (
            raw_expected_subdir / f"{RawFeatureFileNames.RAW_FEATURES.value}.h5"
        )
        self.assertTrue(raw_expected_file.exists(), "Output file not created")
        self.assertTrue(raw_expected_file.is_file(), "Output is not a file")

        # Export the agg features
        agg_output_subdir_path: Path = self.agg_feature_exporter.export_data(
            self.temp_path, agg_feature_set_entry
        )
        # Verify export succeeded with correct output path
        self.assertIsInstance(agg_output_subdir_path, Path)
        # Verify subdirectory was created with correct name
        agg_expected_subdir = (
            self.temp_path / f"aggregate_features_{self.agg_feature_ID_value}"
        )
        self.assertEqual(str(agg_output_subdir_path), str(agg_expected_subdir))
        self.assertTrue(agg_output_subdir_path.exists(), "Subdirectory not created")
        self.assertTrue(
            agg_output_subdir_path.is_dir(), "Subdirectory is not a directory"
        )
        # Verify file was created with correct name
        agg_expected_file = (
            agg_expected_subdir
            / f"{AggregateFeatureFileNames.AGGREGATE_FEATURES.value}.h5"
        )
        self.assertTrue(agg_expected_file.exists(), "Output file not created")
        self.assertTrue(agg_expected_file.is_file(), "Output is not a file")

        # Import raw and agg features
        raw_feature_set_entry_import = self.raw_feature_importer.import_data(
            raw_expected_subdir
        )
        agg_feature_set_entry_import = self.agg_feature_importer.import_data(
            agg_expected_subdir
        )

        # Assertions to verify the results
        self.assertIsInstance(raw_feature_set_entry_import, RawFeatureSetEntry)
        self.assertIsInstance(agg_feature_set_entry_import, AggregateFeatureSetEntry)
        # Assert raw feature types aggregated correctly
        self.assertEqual(len(raw_feature_set_entry_import.raw_epoch_features), 6)
        self.assertEqual(
            raw_feature_set_entry_import.metadata.imu_data_identifier,
            IMUDataIdentifier("mock_imu_data_id"),
        )
        self.assertEqual(
            raw_feature_set_entry_import.metadata.user_identifier,
            UserIdentifier("mock_user_id"),
        )
        for raw_epoch_feature in raw_feature_set_entry_import.raw_epoch_features:
            self.assertEqual(len(raw_epoch_feature.raw_features), 46)
        # Assert agg feature types computed and aggregated correctly
        self.assertEqual(len(agg_feature_set_entry_import.aggregate_features), 46)
        self.assertEqual(
            raw_feature_set_entry_import.metadata.imu_data_identifier,
            IMUDataIdentifier("mock_imu_data_id"),
        )
        self.assertEqual(
            raw_feature_set_entry_import.metadata.user_identifier,
            UserIdentifier("mock_user_id"),
        )
        for agg_feature in agg_feature_set_entry_import.aggregate_features:
            self.assertEqual(len(agg_feature.descriptive_statistics), 9)


if __name__ == "__main__":
    unittest.main()
