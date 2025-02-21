import numpy as np

from src.data_io.builders.model_builders.features.raw.raw_feature_set_entry_builder import (
    RawFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from test.base_test import BaseTest


class TestRawFeatureBuilder(BaseTest):
    def setUp(self):
        self.builder = RawFeatureSetEntryBuilder()

    # ... rest of test code ...


if __name__ == "__main__":
    TestRawFeatureBuilder.run_tests()
