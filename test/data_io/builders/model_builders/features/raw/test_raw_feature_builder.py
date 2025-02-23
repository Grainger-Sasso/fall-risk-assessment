import numpy as np

from src.data_io.builders.model_builders.features.raw.raw_feature_set_entry_builder import (
    RawFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_epoch_features import RawEpochFeatures
from src.data_model.features.raw.raw_feature import RawFeature
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FeatureDataHelper, TestConstants


class TestRawFeatureBuilder(BaseTest):
    def setUp(self):
        self.builder = RawFeatureSetEntryBuilder()
        self.data_helper = FeatureDataHelper()

    def test_build_valid_data(self):
        # Create test HDF5 group with feature data
        input_group: HDF5Group = self.data_helper.create_test_raw_feature_hdf5()
        # Test building feature data
        result: RawFeatureSetEntry = self.builder.build(input_group)

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
                TestConstants.EPOCH_START_TIMES.value[epoch_ix]
                + TestConstants.RAW_FEATURE_EPOCH_LEN.value,
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
                self.assertEqual(raw_feature.value, TestConstants.FEATURE_DATA.value[epoch_ix][feat_ix])

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestRawFeatureBuilder.run_tests()
