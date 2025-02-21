import unittest

from src.data_io.builders.file_builders.feature.aggregate.aggregate_feature_file_builder import (
    AggregateFeatureSetEntryFileBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from test.data_io.test_data.test_data_helper import FeatureDataHelper


class TestAggregateFeatureFileBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = AggregateFeatureSetEntryFileBuilder()
        self.data_helper = FeatureDataHelper()

    def test_build_valid_feature(self):
        # TODO: Create test feature data
        feature_data: AggregateFeatureSetEntry = self.data_helper.create_test_aggregate_feature()
        # Test building HDF5 group
        result: HDF5Group = self.builder.build(feature_data)

        # Assertions
        self.assertIsInstance(result, HDF5Group)
        # self.assertEqual()

    def test_build_empty_feature(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    unittest.main()
