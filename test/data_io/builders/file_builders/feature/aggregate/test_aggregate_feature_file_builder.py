from src.data_io.builders.file_builders.feature.aggregate.aggregate_feature_file_builder import (
    AggregateFeatureSetEntryFileBuilder,
)
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.aggregate.aggregate_feature_fields import (
    AggregateFeatureFields,
)
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FeatureDataHelper, TestConstants


class TestAggregateFeatureFileBuilder(BaseTest):
    def setUp(self):
        self.builder = AggregateFeatureSetEntryFileBuilder()
        self.data_helper = FeatureDataHelper()

    def test_build_valid_feature(self):
        # Create test feature data
        feature_data: AggregateFeatureSetEntry = (
            self.data_helper.create_test_aggregate_feature()
        )
        # Test building HDF5 group
        result: HDF5Group = self.builder.build(feature_data)

        # Assertions
        self.assertIsInstance(result, HDF5Group)
        self.assertEqual(result.name, AggregateFeatureFields.AGGREGATE_FEATURE.value)
        # Test parent group attributes
        agg_feature_attr = result.attributes
        self.assertIsInstance(agg_feature_attr, dict)
        self.assertIn(
            AggregateFeatureFields.AGGREGATE_FEATURE_IDENTIFIER.value,
            agg_feature_attr.keys(),
        )
        self.assertIn(
            AggregateFeatureFields.RAW_FEATURE_IDENTIFIER.value, agg_feature_attr.keys()
        )
        self.assertIn(
            AggregateFeatureFields.IMU_DATA_IDENTIFIER.value, agg_feature_attr.keys()
        )
        self.assertIn(
            AggregateFeatureFields.USER_DATA_IDENTIFIER.value, agg_feature_attr.keys()
        )
        self.assertEqual(
            agg_feature_attr[AggregateFeatureFields.AGGREGATE_FEATURE_IDENTIFIER.value],
            TestConstants.AGG_FEATURE_ID.value,
        )
        self.assertEqual(
            agg_feature_attr[AggregateFeatureFields.RAW_FEATURE_IDENTIFIER.value],
            TestConstants.RAW_FEATURE_ID.value,
        )
        self.assertEqual(
            agg_feature_attr[AggregateFeatureFields.IMU_DATA_IDENTIFIER.value],
            TestConstants.FEATURE_IMU_DATA_ID.value,
        )
        self.assertEqual(
            agg_feature_attr[AggregateFeatureFields.USER_DATA_IDENTIFIER.value],
            TestConstants.FEATURE_USER_DATA_ID.value,
        )
        # Assert parent group items: data, row names, col names
        agg_feature_items = result.items
        agg_feature_item_names = [item.name for item in agg_feature_items]
        self.assertIn(AggregateFeatureFields.FEATURES.value, agg_feature_item_names)
        self.assertIn(
            AggregateFeatureFields.FEATURE_NAMES.value, agg_feature_item_names
        )
        self.assertIn(
            AggregateFeatureFields.DESCRIPTIVE_STATISTIC_NAMES.value,
            agg_feature_item_names,
        )
        feature_names: HDF5Dataset = result.get_item_by_name(
            AggregateFeatureFields.FEATURE_NAMES.value
        )
        self.assertEqual(
            feature_names.data,
            [RawFeatureType.PLACEHOLDER.value, RawFeatureType.PLACEHOLDER.value],
        )
        stat_names: HDF5Dataset = result.get_item_by_name(
            AggregateFeatureFields.DESCRIPTIVE_STATISTIC_NAMES.value
        )
        self.assertEqual(
            stat_names.data,
            [
                DescriptiveStatisticType.PLACEHOLDER.value,
                DescriptiveStatisticType.PLACEHOLDER.value,
            ],
        )
        features: HDF5Dataset = result.get_item_by_name(
            AggregateFeatureFields.FEATURES.value
        )
        self.assertEqual(
            features.data,
            [
                [
                    TestConstants.PLACEHOLDER_STAT_VALUE.value,
                    TestConstants.PLACEHOLDER_STAT_VALUE.value + 1.0,
                ],
                [
                    TestConstants.PLACEHOLDER_STAT_VALUE.value,
                    TestConstants.PLACEHOLDER_STAT_VALUE.value + 1.0,
                ],
            ],
        )

    def test_build_empty_feature(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestAggregateFeatureFileBuilder.run_tests()
