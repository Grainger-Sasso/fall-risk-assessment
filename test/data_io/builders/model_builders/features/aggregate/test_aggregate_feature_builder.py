import numpy as np

from src.data_io.builders.model_builders.features.aggregate.aggregate_feature_set_entry_builder import (
    AggregateFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from test.base_test import BaseTest


class TestAggregateFeatureBuilder(BaseTest):
    def setUp(self):
        self.builder = AggregateFeatureSetEntryBuilder()

    def test_build_valid_data(self):
        # TODO: Create test HDF5 group with feature data
        input_group = HDF5Group()

        # Test building feature data
        result = self.builder.build(input_group)

        # Assertions
        self.assertIsInstance(result, AggregateFeatureSetEntry)
        # TODO: Add more specific assertions

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestAggregateFeatureBuilder.run_tests()
