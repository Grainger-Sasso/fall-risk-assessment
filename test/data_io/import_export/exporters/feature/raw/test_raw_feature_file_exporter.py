import os
import tempfile
from pathlib import Path

import numpy as np  # type: ignore

from src.data_io.builders.file_builders.feature.raw.raw_feature_file_builder import (
    RawFeatureSetEntryFileBuilder,
)
from src.data_io.builders.model_builders.features.raw.raw_feature_set_entry_builder import (
    RawFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.exporters.feature.raw.raw_feature_file_exporter import (
    RawFeatureFileExporter,
    RawFeatureFileNames,
)
from src.data_io.import_export.importers.features.raw.raw_feature_importer import (
    RawFeatureFileNames,
    RawFeatureImporter,
)
from src.data_io.model_fields.features.raw.raw_feature_fields import (
    RawFeatureFields,
)
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_epoch_features import RawEpochFeatures
from src.data_model.features.raw.raw_feature import RawFeature
from src.data_model.features.raw.raw_feature_set_entry import (
    RawFeatureSetEntry,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.raw_feature_identifier import (
    RawFeatureIdentifier,
)
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import (
    FeatureDataHelper,
    TestConstants,
)


class TestRawFeatureFileExporter(BaseTest):
    def setUp(self):
        # Create test objects
        self.exporter = RawFeatureFileExporter()
        self.importer = RawFeatureImporter()
        self.helper = FeatureDataHelper()

        # Create temp directory for test
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test data
        self.test_data = self.helper.create_test_raw_feature()

    def tearDown(self):
        # Clean up test directory
        if self.temp_path.exists():
            # Walk directory tree bottom-up to properly remove all contents
            for root, dirs, files in os.walk(self.temp_path, topdown=False):
                # First remove all files in current directory
                for name in files:
                    (Path(root) / name).unlink()
                # Then remove the directory itself
                for name in dirs:
                    (Path(root) / name).rmdir()
            # Finally remove the temp directory
            self.temp_path.rmdir()

    def test_export_data(self):
        # Export the test data
        output_subdir_path: Path = self.exporter.export_data(
            self.temp_path, self.test_data
        )

        # Verify export succeeded with correct output path
        self.assertIsInstance(output_subdir_path, Path)
        # Verify subdirectory was created with correct name
        expected_subdir = (
            self.temp_path / f"raw_features_{TestConstants.RAW_FEATURE_ID.value}"
        )
        self.assertEqual(str(output_subdir_path), str(expected_subdir))

        self.assertTrue(expected_subdir.exists(), "Subdirectory not created")
        self.assertTrue(expected_subdir.is_dir(), "Subdirectory is not a directory")

        # Verify file was created with correct name
        expected_file = expected_subdir / f"{RawFeatureFileNames.RAW_FEATURES.value}.h5"
        self.assertTrue(expected_file.exists(), "Output file not created")
        self.assertTrue(expected_file.is_file(), "Output is not a file")

        # Import the exported data
        result = self.importer.import_data(expected_subdir)

        # Assert raw feature set entry
        self.assertIsInstance(result, RawFeatureSetEntry)
        # Assert metadata
        metadata = result.metadata
        self.assertIsInstance(metadata, RawFeatureSetEntryMetadata)
        self.assertIsInstance(metadata.raw_feature_identifier, RawFeatureIdentifier)
        self.assertIsInstance(metadata.user_identifier, UserIdentifier)
        self.assertIsInstance(metadata.imu_data_identifier, IMUDataIdentifier)
        self.assertIsInstance(metadata.start_time, float)
        self.assertIsInstance(metadata.epoch_length, float)
        self.assertEqual(
            metadata.raw_feature_identifier.value,
            TestConstants.RAW_FEATURE_ID.value,
        )
        self.assertEqual(
            metadata.user_identifier.value, TestConstants.FEATURE_USER_DATA_ID.value
        )
        self.assertEqual(
            metadata.imu_data_identifier.value,
            TestConstants.FEATURE_IMU_DATA_ID.value,
        )
        self.assertEqual(
            metadata.start_time, TestConstants.RAW_FEATURE_START_TIME.value
        )
        self.assertEqual(
            metadata.epoch_length, TestConstants.RAW_FEATURE_EPOCH_LEN.value
        )

        # Assert raw feature data
        epoch_feature_list = result.raw_epoch_features
        self.assertIsInstance(epoch_feature_list, list)
        self.assertEqual(len(epoch_feature_list), 2)

        feature_0 = epoch_feature_list[0]
        self.assertIsInstance(feature_0, RawEpochFeatures)
        self.assertEqual(
            feature_0.epoch_start_time,
            TestConstants.RAW_FEATURE_START_TIME.value,
        )
        self.assertEqual(
            feature_0.epoch_end_time,
            TestConstants.RAW_FEATURE_END_TIME.value,
        )
        self.assertIsInstance(feature_0.raw_features, list)
        self.assertEqual(
            len(feature_0.raw_features),
            2,
        )
        self.assertIsInstance(feature_0.raw_features[0], RawFeature)
        self.assertIsInstance(feature_0.raw_features[0].feature_type, RawFeatureType)
        self.assertEqual(
            feature_0.raw_features[0].feature_type.value,
            RawFeatureType.DAY_N.value,
        )
        self.assertIsInstance(feature_0.raw_features[0].value, float)
        self.assertEqual(
            feature_0.raw_features[0].value,
            TestConstants.PLACEHOLDER_FEATURE_VALUE.value,
        )
        self.assertIsInstance(feature_0.raw_features[1], RawFeature)
        self.assertIsInstance(feature_0.raw_features[1].feature_type, RawFeatureType)
        self.assertEqual(
            feature_0.raw_features[1].feature_type.value,
            RawFeatureType.DAY_N.value,
        )
        self.assertIsInstance(feature_0.raw_features[1].value, float)
        self.assertEqual(
            feature_0.raw_features[1].value,
            TestConstants.PLACEHOLDER_FEATURE_VALUE.value + 1.0,
        )

        feature_1 = epoch_feature_list[1]
        self.assertIsInstance(feature_1, RawEpochFeatures)
        self.assertEqual(
            feature_1.epoch_start_time,
            TestConstants.RAW_FEATURE_START_TIME.value + 0.1,
        )
        self.assertEqual(
            feature_1.epoch_end_time,
            TestConstants.RAW_FEATURE_END_TIME.value + 0.1,
        )
        self.assertIsInstance(feature_1.raw_features, list)
        self.assertEqual(
            len(feature_1.raw_features),
            2,
        )
        self.assertIsInstance(feature_1.raw_features[0], RawFeature)
        self.assertIsInstance(feature_1.raw_features[0].feature_type, RawFeatureType)
        self.assertEqual(
            feature_1.raw_features[0].feature_type.value,
            RawFeatureType.DAY_N.value,
        )
        self.assertIsInstance(feature_1.raw_features[0].value, float)
        self.assertEqual(
            feature_1.raw_features[0].value,
            TestConstants.PLACEHOLDER_FEATURE_VALUE.value,
        )
        self.assertIsInstance(feature_1.raw_features[1], RawFeature)
        self.assertIsInstance(feature_1.raw_features[1].feature_type, RawFeatureType)
        self.assertEqual(
            feature_1.raw_features[1].feature_type.value,
            RawFeatureType.DAY_N.value,
        )
        self.assertIsInstance(feature_1.raw_features[1].value, float)
        self.assertEqual(
            feature_1.raw_features[1].value,
            TestConstants.PLACEHOLDER_FEATURE_VALUE.value + 1.0,
        )

    def test_export_data_directory_creation_fails(self):
        # Create a file with the same name as our intended subdirectory to cause creation to fail
        conflict_path = (
            self.temp_path / f"raw_features_{TestConstants.RAW_FEATURE_ID.value}"
        )
        conflict_path.touch()

        with self.assertRaises(Exception) as context:
            self.exporter.export_data(self.temp_path, self.test_data)

        self.assertIn("Failed to create directory", str(context.exception))

        # Clean up
        conflict_path.unlink()

    def test_export_data_write_fails(self):
        # Create directory structure first with write permissions
        readonly_dir = self.temp_path / "readonly"
        target_dir = readonly_dir / f"raw_features_{TestConstants.RAW_FEATURE_ID.value}"
        target_dir.mkdir(parents=True)

        # Then make parent directory read-only
        readonly_dir.chmod(0o444)

        with self.assertRaises(Exception) as context:
            self.exporter.export_data(readonly_dir, self.test_data)

        self.assertIn("Permission denied", str(context.exception))

        # Clean up - restore permissions to allow deletion
        readonly_dir.chmod(0o777)


if __name__ == "__main__":
    TestRawFeatureFileExporter.run_tests()
