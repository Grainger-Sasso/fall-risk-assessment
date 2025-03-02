import tempfile
from pathlib import Path

import numpy as np # type: ignore

from src.data_io.builders.file_builders.feature.aggregate.aggregate_feature_file_builder import (
    AggregateFeatureSetEntryFileBuilder,
)
from src.data_io.builders.model_builders.features.aggregate.aggregate_feature_set_entry_builder import (
    AggregateFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.importers.features.aggregate.aggregate_feature_importer import (
    AggregateFeatureFileNames,
    AggregateFeatureImporter,
)
from src.data_io.model_fields.features.aggregate.aggregate_feature_fields import (
    AggregateFeatureFields,
)
from src.data_model.features.aggregate.aggregate_feature import AggregateFeature
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.aggregate.descriptive_statistic import DescriptiveStatistic
from src.data_model.features.aggregate.metadata.aggregate_feature_set_entry_metadata import (
    AggregateFeatureSetEntryMetadata,
)
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import (
    FeatureDataHelper,
    TestConstants,
)


class TestAggregateFeatureImporter(BaseTest):
    def setUp(self):
        self.importer = AggregateFeatureImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = FeatureDataHelper()

        # Create test file in temp directory
        self.feature_path = self.helper.create_test_aggregate_feature_file(
            self.temp_path / f"{AggregateFeatureFileNames.AGGREGATE_FEATURES.value}.h5"
        )

    def tearDown(self):
        # Clean up test files
        if self.feature_path.exists():
            self.feature_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Assertions
        self.assertIsInstance(result, AggregateFeatureSetEntry)
        # Assert the metadata
        agg_metadata = result.metadata
        self.assertIsInstance(agg_metadata, AggregateFeatureSetEntryMetadata)
        self.assertIsInstance(
            agg_metadata.aggregate_feature_identifier, AggregateFeatureIdentifier
        )
        self.assertIsInstance(agg_metadata.raw_feature_identifier, RawFeatureIdentifier)
        self.assertIsInstance(agg_metadata.user_identifier, UserIdentifier)
        self.assertIsInstance(agg_metadata.imu_data_identifier, IMUDataIdentifier)
        self.assertEqual(
            agg_metadata.aggregate_feature_identifier.value,
            TestConstants.AGG_FEATURE_ID.value,
        )
        self.assertEqual(
            agg_metadata.raw_feature_identifier.value,
            TestConstants.RAW_FEATURE_ID.value,
        )
        self.assertEqual(
            agg_metadata.user_identifier.value, TestConstants.FEATURE_USER_DATA_ID.value
        )
        self.assertEqual(
            agg_metadata.imu_data_identifier.value,
            TestConstants.FEATURE_IMU_DATA_ID.value,
        )

        # Assert aggregate feature data
        agg_feature_list = result.aggregate_features
        self.assertIsInstance(agg_feature_list, list)
        self.assertEqual(
            len(agg_feature_list), len(TestConstants.RAW_FEATURE_NAMES.value)
        )

        for agg_ix, agg_feature in enumerate(agg_feature_list):
            self.assertIsInstance(agg_feature, AggregateFeature)
            self.assertIsInstance(agg_feature.feature_type, RawFeatureType)
            self.assertEqual(
                agg_feature.feature_type.value,
                TestConstants.RAW_FEATURE_NAMES.value[agg_ix],
            )
            self.assertIsInstance(agg_feature.descriptive_statistics, list)
            self.assertEqual(
                len(agg_feature.descriptive_statistics),
                len(TestConstants.STAT_NAMES.value),
            )
            for stat_ix, stat in enumerate(agg_feature.descriptive_statistics):
                self.assertIsInstance(stat, DescriptiveStatistic)
                self.assertIsInstance(stat.statistic_type, DescriptiveStatisticType)
                self.assertEqual(
                    stat.statistic_type.value, TestConstants.STAT_NAMES.value[stat_ix]
                )
                self.assertIsInstance(stat.value, float)
                self.assertEqual(
                    stat.value, TestConstants.FEATURE_DATA.value[agg_ix][stat_ix]
                )

    def test_missing_file(self):
        # Remove the required file
        self.feature_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestAggregateFeatureImporter.run_tests()
