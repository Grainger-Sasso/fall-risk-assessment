import tempfile
from pathlib import Path

from src.data_io.import_export.importers.mapping.mapping_importer import (
    MappingFileNames,
    MappingImporter,
)
from src.database_manager.mapping.mapping import Mapping
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import MappingHelper, TestConstants


class TestMappingImporter(BaseTest):
    def setUp(self):
        self.importer = MappingImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = MappingHelper()

        # Create test file in temp directory
        self.map_path = self.helper.create_test_mapping_file(
            self.temp_path / f"{MappingFileNames.MAPPING.value}.csv",
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
        self.assertIsInstance(result, Mapping)
        self.assertEqual(len(result.map), len(TestConstants.MAPPING_SOURCE_IDS.value))

        # Test mappings
        for ix, (source_id, target_id) in enumerate(result.map.items()):
            self.assertIsInstance(source_id, str)
            self.assertIsInstance(target_id, str)
            self.assertEqual(source_id, TestConstants.MAPPING_SOURCE_IDS.value[ix])
            self.assertEqual(target_id, TestConstants.MAPPING_TARGET_IDS.value[ix])

    def test_missing_file(self):
        # Remove the required file
        self.map_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestMappingImporter.run_tests()
