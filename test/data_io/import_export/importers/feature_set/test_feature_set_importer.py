import tempfile
from pathlib import Path

from src.data_io.import_export.importers.feature_set.feature_set_importer import (
    FeatureSetFileNames,
    FeatureSetImporter,
)
from src.data_model.feature_set.feature_set import FeatureSet
from src.data_model.feature_set.feature_set_entry import FeatureSetEntry
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FeatureSetHelper, TestConstants


class TestFeatureSetImporter(BaseTest):
    def setUp(self):
        self.importer = FeatureSetImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = FeatureSetHelper()

        # Create test file in temp directory
        self.feature_set_path = self.helper.create_test_feature_set_file(
            self.temp_path / f"{FeatureSetFileNames.FEATURE_SET.value}.csv"
        )

    def tearDown(self):
        # Clean up test files
        if self.feature_set_path.exists():
            self.feature_set_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(
            self.temp_path, TestConstants.FEATURE_SET_NAME.value
        )

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

    def test_missing_file(self):
        # Remove the required file
        self.feature_set_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path, "")


if __name__ == "__main__":
    TestFeatureSetImporter.run_tests()
