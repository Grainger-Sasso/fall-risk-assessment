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


class TestRawFeatureImporter(BaseTest):
    def setUp(self):
        self.importer = RawFeatureImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = FeatureDataHelper()

        # Create test file in temp directory
        self.feature_path = self.helper.create_test_raw_feature_file(
            self.temp_path / f"{RawFeatureFileNames.RAW_FEATURES.value}.h5"
        )

    def tearDown(self):
        # Clean up test files
        if self.feature_path.exists():
            self.feature_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

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
        self.assertEqual(
            len(epoch_feature_list), len(TestConstants.EPOCH_START_TIMES.value)
        )

        for epoch_ix, epoch_feature in enumerate(epoch_feature_list):
            self.assertIsInstance(epoch_feature, RawEpochFeatures)
            self.assertEqual(
                epoch_feature.epoch_start_time,
                TestConstants.EPOCH_START_TIMES.value[epoch_ix],
            )
            self.assertEqual(
                epoch_feature.epoch_end_time,
                TestConstants.EPOCH_END_TIMES.value[epoch_ix],
            )
            self.assertIsInstance(epoch_feature.raw_features, list)
            self.assertEqual(
                len(epoch_feature.raw_features),
                len(TestConstants.RAW_FEATURE_NAMES.value),
            )
            for feat_ix, raw_feature in enumerate(epoch_feature.raw_features):
                self.assertIsInstance(raw_feature, RawFeature)
                self.assertIsInstance(raw_feature.feature_type, RawFeatureType)
                self.assertEqual(
                    raw_feature.feature_type.value,
                    TestConstants.RAW_FEATURE_NAMES.value[feat_ix],
                )
                self.assertIsInstance(raw_feature.value, float)
                self.assertEqual(
                    raw_feature.value,
                    TestConstants.FEATURE_DATA.value[epoch_ix][feat_ix],
                )

    def test_missing_file(self):
        # Remove the required file
        self.feature_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestRawFeatureImporter.run_tests()
