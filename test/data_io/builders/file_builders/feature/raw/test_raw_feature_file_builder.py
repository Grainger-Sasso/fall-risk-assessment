from src.data_io.builders.file_builders.feature.raw.raw_feature_file_builder import (
    RawFeatureSetEntryFileBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FeatureDataHelper


class TestRawFeatureFileBuilder(BaseTest):
    def setUp(self):
        self.builder = RawFeatureSetEntryFileBuilder()
        self.data_helper = FeatureDataHelper()

    def test_build_valid_feature(self):
        # TODO: Create test feature data
        feature_data = RawFeatureSetEntry([], None)

        # Test building HDF5 group
        result = self.builder.build(feature_data)

        # Assertions
        self.assertIsInstance(result, HDF5Group)
        # TODO: Add more specific assertions

    def test_build_empty_feature(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestRawFeatureFileBuilder.run_tests()
