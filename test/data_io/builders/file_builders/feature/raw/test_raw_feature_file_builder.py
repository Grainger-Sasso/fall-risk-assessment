from src.data_io.builders.file_builders.feature.raw.raw_feature_file_builder import (
    RawFeatureSetEntryFileBuilder,
)
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.raw.raw_feature_fields import RawFeatureFields
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_epoch_features import RawEpochFeatures
from src.data_model.features.raw.raw_feature import RawFeature
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_types.feature.raw_feature_type import RawFeatureType
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FeatureDataHelper, TestConstants


class TestRawFeatureFileBuilder(BaseTest):
    def setUp(self):
        self.builder = RawFeatureSetEntryFileBuilder()
        self.data_helper = FeatureDataHelper()

    def test_build_valid_feature(self):
        # Create test feature data
        feature_data: RawFeatureSetEntry = self.data_helper.create_test_raw_feature()
        # Test building HDF5 group
        result: HDF5Group = self.builder.build(feature_data)

        # Assertions
        self.assertIsInstance(result, HDF5Group)
        self.assertEqual(result.name, RawFeatureFields.RAW_FEATURE.value)
        # Assert parent group attributes
        attributes = result.attributes
        self.assertIsInstance(attributes, dict)
        self.assertIn(RawFeatureFields.RAW_FEATURE_IDENTIFIER.value, attributes.keys())
        self.assertIn(RawFeatureFields.USER_DATA_IDENTIFIER.value, attributes.keys())
        self.assertIn(RawFeatureFields.IMU_DATA_IDENTIFIER.value, attributes.keys())
        self.assertIn(RawFeatureFields.EPOCH_LEN.value, attributes.keys())
        self.assertIn(RawFeatureFields.START_TIME.value, attributes.keys())
        self.assertEqual(
            attributes[RawFeatureFields.RAW_FEATURE_IDENTIFIER.value],
            TestConstants.RAW_FEATURE_ID.value,
        )
        self.assertEqual(
            attributes[RawFeatureFields.USER_DATA_IDENTIFIER.value],
            TestConstants.FEATURE_USER_DATA_ID.value,
        )
        self.assertEqual(
            attributes[RawFeatureFields.IMU_DATA_IDENTIFIER.value],
            TestConstants.FEATURE_IMU_DATA_ID.value,
        )
        self.assertEqual(
            attributes[RawFeatureFields.EPOCH_LEN.value],
            TestConstants.RAW_FEATURE_EPOCH_LEN.value,
        )
        self.assertEqual(
            attributes[RawFeatureFields.START_TIME.value],
            TestConstants.RAW_FEATURE_START_TIME.value,
        )
        # Assert parent group items
        feature_items = result.items
        self.assertIsInstance(feature_items, list)
        self.assertEqual(len(feature_items), 4)
        features = result.get_item_by_name(RawFeatureFields.FEATURES.value)
        self.assertIsInstance(features, HDF5Dataset)
        self.assertEqual(
            features.data,
            [
                [
                    TestConstants.PLACEHOLDER_FEATURE_VALUE.value,
                    TestConstants.PLACEHOLDER_FEATURE_VALUE.value + 1.0,
                ],
                [
                    TestConstants.PLACEHOLDER_FEATURE_VALUE.value,
                    TestConstants.PLACEHOLDER_FEATURE_VALUE.value + 1.0,
                ],
            ],
        )
        self.assertEqual(features.attributes, {})

        feature_names = result.get_item_by_name(RawFeatureFields.FEATURE_NAMES.value)
        self.assertIsInstance(feature_names, HDF5Dataset)
        self.assertEqual(
            feature_names.data,
            [RawFeatureType.PLACEHOLDER.value, RawFeatureType.PLACEHOLDER.value],
        )
        self.assertEqual(feature_names.attributes, {})

        epoch_starts = result.get_item_by_name(RawFeatureFields.EPOCH_STARTS.value)
        self.assertIsInstance(epoch_starts, HDF5Dataset)
        self.assertEqual(
            epoch_starts.data,
            [
                TestConstants.RAW_FEATURE_START_TIME.value,
                TestConstants.RAW_FEATURE_START_TIME.value + 0.1,
            ],
        )
        self.assertEqual(epoch_starts.attributes, {})

        epoch_ends = result.get_item_by_name(RawFeatureFields.EPOCH_ENDS.value)
        self.assertIsInstance(epoch_ends, HDF5Dataset)
        self.assertEqual(
            epoch_ends.data,
            [
                TestConstants.RAW_FEATURE_END_TIME.value,
                TestConstants.RAW_FEATURE_END_TIME.value + 0.1,
            ],
        )
        self.assertEqual(epoch_ends.attributes, {})

    def test_build_empty_feature(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestRawFeatureFileBuilder.run_tests()
