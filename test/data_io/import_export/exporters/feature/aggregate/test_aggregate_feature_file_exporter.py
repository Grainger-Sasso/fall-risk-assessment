import os
import tempfile
from pathlib import Path

import numpy as np  # type: ignore

from src.data_io.builders.file_builders.feature.aggregate.aggregate_feature_file_builder import (
    AggregateFeatureSetEntryFileBuilder,
)
from src.data_io.builders.model_builders.features.aggregate.aggregate_feature_set_entry_builder import (
    AggregateFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.exporters.feature.aggregate.aggregate_feature_file_exporter import (
    AggregateFeatureFileExporter,
    AggregateFeatureFileNames,
)
from src.data_io.import_export.importers.features.aggregate.aggregate_feature_importer import (
    AggregateFeatureFileNames,
    AggregateFeatureImporter,
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
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import (
    FeatureDataHelper,
    TestConstants,
)


class TestAggregateFeatureFileExporter(BaseTest):
    def setUp(self):
        # Create test objects
        self.exporter = AggregateFeatureFileExporter()
        self.importer = AggregateFeatureImporter()
        self.helper = FeatureDataHelper()

        # Create temp directory for test
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test data
        self.test_data = self.helper.create_test_aggregate_feature()

    def tearDown(self):
        # Clean up test directory
        if self.temp_path.exists():
            # Walk directory tree bottom-up to properly remove all contents
            for root, dirs, files in os.walk(self.temp_path, topdown=False):
                # First remove all files in current directory
                for name in files:
                    (Path(root) / name).unlink()
                # Then remove the directory itself
                for name in dirs:
                    (Path(root) / name).rmdir()
            # Finally remove the temp directory
            self.temp_path.rmdir()

    def test_export_data(self):
        # Export the test data
        output_subdir_path: Path = self.exporter.export_data(
            self.temp_path, self.test_data
        )

        # Verify export succeeded with correct output path
        self.assertIsInstance(output_subdir_path, Path)
        # Verify subdirectory was created with correct name
        expected_subdir = (
            self.temp_path / f"aggregate_features_{TestConstants.AGG_FEATURE_ID.value}"
        )
        self.assertEqual(str(output_subdir_path), str(expected_subdir))

        self.assertTrue(expected_subdir.exists(), "Subdirectory not created")
        self.assertTrue(expected_subdir.is_dir(), "Subdirectory is not a directory")

        # Verify file was created with correct name
        expected_file = (
            expected_subdir / f"{AggregateFeatureFileNames.AGGREGATE_FEATURES.value}.h5"
        )
        self.assertTrue(expected_file.exists(), "Output file not created")
        self.assertTrue(expected_file.is_file(), "Output is not a file")

        # Import the exported data
        result = self.importer.import_data(expected_subdir)

        # Assert aggregate feature set entry
        self.assertIsInstance(result, AggregateFeatureSetEntry)

        # Assert metadata
        metadata = result.metadata
        self.assertIsInstance(metadata, AggregateFeatureSetEntryMetadata)
        self.assertIsInstance(
            metadata.aggregate_feature_identifier, AggregateFeatureIdentifier
        )
        self.assertIsInstance(metadata.raw_feature_identifier, RawFeatureIdentifier)
        self.assertEqual(
            metadata.aggregate_feature_identifier.value,
            TestConstants.AGG_FEATURE_ID.value,
        )
        self.assertEqual(
            metadata.raw_feature_identifier.value,
            TestConstants.RAW_FEATURE_ID.value,
        )

        # Assert aggregate features
        self.assertIsInstance(result.aggregate_features, list)
        self.assertEqual(len(result.aggregate_features), 2)

        # Test first feature
        feature_0 = result.aggregate_features[0]
        self.assertIsInstance(feature_0, AggregateFeature)
        self.assertIsInstance(feature_0.feature_type, RawFeatureType)
        self.assertEqual(
            feature_0.feature_type.value,
            RawFeatureType.DAY_N.value,
        )
        self.assertIsInstance(feature_0.descriptive_statistics, list)
        self.assertEqual(
            len(feature_0.descriptive_statistics),
            2,
        )
        self.assertIsInstance(feature_0.descriptive_statistics[0], DescriptiveStatistic)
        self.assertIsInstance(
            feature_0.descriptive_statistics[0].statistic_type, DescriptiveStatisticType
        )
        self.assertEqual(
            feature_0.descriptive_statistics[0].value,
            TestConstants.PLACEHOLDER_STAT_VALUE.value,
        )
        self.assertIsInstance(feature_0.descriptive_statistics[1], DescriptiveStatistic)
        self.assertIsInstance(
            feature_0.descriptive_statistics[1].statistic_type, DescriptiveStatisticType
        )
        self.assertEqual(
            feature_0.descriptive_statistics[1].value,
            TestConstants.PLACEHOLDER_STAT_VALUE.value + 1.0,
        )

        # Test second feature
        feature_1 = result.aggregate_features[1]
        self.assertIsInstance(feature_1, AggregateFeature)
        self.assertIsInstance(feature_1.feature_type, RawFeatureType)
        self.assertEqual(
            feature_1.feature_type.value,
            RawFeatureType.DAY_N.value,
        )
        self.assertIsInstance(feature_1.descriptive_statistics, list)
        self.assertEqual(
            len(feature_1.descriptive_statistics),
            2,
        )
        self.assertIsInstance(feature_1.descriptive_statistics[0], DescriptiveStatistic)
        self.assertIsInstance(
            feature_1.descriptive_statistics[0].statistic_type, DescriptiveStatisticType
        )
        self.assertEqual(
            feature_1.descriptive_statistics[0].value,
            TestConstants.PLACEHOLDER_STAT_VALUE.value,
        )
        self.assertIsInstance(feature_1.descriptive_statistics[1], DescriptiveStatistic)
        self.assertIsInstance(
            feature_1.descriptive_statistics[1].statistic_type, DescriptiveStatisticType
        )
        self.assertEqual(
            feature_1.descriptive_statistics[1].value,
            TestConstants.PLACEHOLDER_STAT_VALUE.value + 1.0,
        )

    def test_export_data_directory_creation_fails(self):
        # Create a file with the same name as our intended subdirectory to cause creation to fail
        conflict_path = (
            self.temp_path / f"aggregate_features_{TestConstants.AGG_FEATURE_ID.value}"
        )
        conflict_path.touch()

        with self.assertRaises(Exception) as context:
            self.exporter.export_data(self.temp_path, self.test_data)

        self.assertIn(
            "Export failed: Failed to create directory at", str(context.exception)
        )

        # Clean up
        conflict_path.unlink()

    def test_export_data_write_fails(self):
        # Create directory structure first with write permissions
        readonly_dir = self.temp_path / "readonly"
        target_dir = (
            readonly_dir / f"aggregate_features_{TestConstants.AGG_FEATURE_ID.value}"
        )
        target_dir.mkdir(parents=True)

        # Then make parent directory read-only
        readonly_dir.chmod(0o444)

        with self.assertRaises(Exception) as context:
            self.exporter.export_data(readonly_dir, self.test_data)

        self.assertIn("Permission denied", str(context.exception))

        # Clean up - restore permissions to allow deletion
        readonly_dir.chmod(0o777)


if __name__ == "__main__":
    TestAggregateFeatureFileExporter.run_tests()
