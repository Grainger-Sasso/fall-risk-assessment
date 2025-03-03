import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.data_io.import_export.importers.importer import Importer
from src.database_manager.data_access.data_loader import DataLoader
from test.base_test import BaseTest


class TestDataLoader(BaseTest):
    def setUp(self):
        # Create mock importer
        self.mock_importer = Mock(spec=Importer)
        self.loader = DataLoader(self.mock_importer)

        # Create temp test directory and file
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.test_file = self.temp_path / "test_file.h5"
        self.test_file.touch()

    def tearDown(self):
        # Clean up test files
        if self.test_file.exists():
            self.test_file.unlink()
        if self.temp_path.exists():
            self.temp_path.rmdir()

    def test_load_success(self):
        # Configure mock to return test data
        expected_data = "test_data"
        self.mock_importer.import_data.return_value = expected_data

        # Test successful load
        result = self.loader.load(self.test_file)
        self.assertEqual(result, expected_data)
        self.mock_importer.import_data.assert_called_once_with(self.temp_path)

    def test_load_file_not_found(self):
        # Test with nonexistent file
        nonexistent_path = self.temp_path / "nonexistent.h5"
        with self.assertRaises(FileNotFoundError):
            self.loader.load(nonexistent_path)

    def test_load_import_error(self):
        # Configure mock to raise exception
        self.mock_importer.import_data.side_effect = Exception("Import failed")

        # Test import failure
        with self.assertRaises(ImportError):
            self.loader.load(self.test_file)


if __name__ == "__main__":
    TestDataLoader.run_tests()
