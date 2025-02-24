from src.data_io.builders.model_builders.mappings.aggregate_id_to_raw_id_map_builder import (
    AggregateIDToRawIDMapBuilder,
)
from src.database_manager.mappings.aggregate_feature_id_to_raw_feature_id_map import (
    AggregateFeatureIDToRawFeatureIDMap,
)
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import MappingHelper, TestConstants


class TestAggregateIDToRawIDMapBuilder(BaseTest):
    def setUp(self):
        self.builder = AggregateIDToRawIDMapBuilder()
        self.data_helper = MappingHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_mapping_csv(
            TestConstants.AGG_TO_RAW_SOURCE_IDS.value,
            TestConstants.AGG_TO_RAW_TARGET_IDS.value,
        )

        # Test building map
        result = self.builder.build(csv_data)

        # Assertions
        self.assertIsInstance(result, AggregateFeatureIDToRawFeatureIDMap)
        self.assertEqual(
            len(result.map), len(TestConstants.AGG_TO_RAW_SOURCE_IDS.value)
        )

        # Test mappings
        for ix, (agg_id, raw_id) in enumerate(result.map.items()):
            self.assertIsInstance(agg_id, AggregateFeatureIdentifier)
            self.assertIsInstance(raw_id, RawFeatureIdentifier)
            self.assertEqual(
                agg_id.value, TestConstants.AGG_TO_RAW_SOURCE_IDS.value[ix]
            )
            self.assertEqual(
                raw_id.value, TestConstants.AGG_TO_RAW_TARGET_IDS.value[ix]
            )

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestAggregateIDToRawIDMapBuilder.run_tests() 