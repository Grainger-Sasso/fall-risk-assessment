from src.data_io.builders.model_builders.feature_set.feature_set_builder import (
    FeatureSetBuilder,
)
from src.data_model.feature_set.feature_set import FeatureSet
from src.data_model.feature_set.feature_set_entry import FeatureSetEntry
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FeatureSetHelper, TestConstants


class TestFeatureSetBuilder(BaseTest):
    def setUp(self):
        self.builder = FeatureSetBuilder()
        self.data_helper = FeatureSetHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_feature_set_csv()

        # Test building feature set
        result = self.builder.build(csv_data, TestConstants.FEATURE_SET_NAME.value)

        # Assertions
        self.assertIsInstance(result, FeatureSet)
        self.assertEqual(result.name, TestConstants.FEATURE_SET_NAME.value)

        # Test entries
        self.assertIsInstance(result.entries, list)
        self.assertEqual(
            len(result.entries), len(TestConstants.FEATURE_SET_RAW_IDS.value)
        )

        for ix, entry in enumerate(result.entries):
            self.assertIsInstance(entry, FeatureSetEntry)
            self.assertIsInstance(entry.raw_feature_identifier, RawFeatureIdentifier)
            self.assertIsInstance(
                entry.aggregate_feature_identifier, AggregateFeatureIdentifier
            )
            self.assertEqual(
                entry.raw_feature_identifier.value,
                TestConstants.FEATURE_SET_RAW_IDS.value[ix],
            )
            self.assertEqual(
                entry.aggregate_feature_identifier.value,
                TestConstants.FEATURE_SET_AGG_IDS.value[ix],
            )

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None, None)


if __name__ == "__main__":
    TestFeatureSetBuilder.run_tests()
