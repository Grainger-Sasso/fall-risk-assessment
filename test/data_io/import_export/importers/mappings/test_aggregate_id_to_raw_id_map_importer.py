import tempfile
from pathlib import Path

from src.data_io.import_export.importers.mappings.aggregate_id_to_raw_id_map_importer import (
    AggregateIDToRawIDMapFileNames,
    AggregateIDToRawIDMapImporter,
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


class TestAggregateIDToRawIDMapImporter(BaseTest):
    def setUp(self):
        self.importer = AggregateIDToRawIDMapImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = MappingHelper()

        # Create test file in temp directory
        self.map_path = self.helper.create_test_mapping_file(
            TestConstants.AGG_TO_RAW_SOURCE_IDS.value,
            TestConstants.AGG_TO_RAW_TARGET_IDS.value,
            self.temp_path
            / f"{AggregateIDToRawIDMapFileNames.AGGREGATE_FEATURE_TO_RAW_FEATURE_ID_MAP.value}.csv",
        )

    def tearDown(self):
        # Clean up test files
        if self.map_path.exists():
            self.map_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

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

    def test_missing_file(self):
        # Remove the required file
        self.map_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestAggregateIDToRawIDMapImporter.run_tests()
