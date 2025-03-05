import tempfile
from pathlib import Path

from src.data_io.import_export.importers.registry.registry_importer import (
    RegistryFileNames,
    RegistryImporter,
)
from src.database_manager.registry.registry import Registry
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestRegistryImporter(BaseTest):
    def setUp(self):
        self.importer = RegistryImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = RegistryHelper()

        # Create test file in temp directory
        self.registry_path = self.helper.create_test_registry_file(
            self.temp_path / f"{RegistryFileNames.REGISTRY.value}.csv",
        )

    def tearDown(self):
        # Clean up test files
        if self.registry_path.exists():
            self.registry_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Assertions
        self.assertIsInstance(result, Registry)
        self.assertEqual(len(result.registry), len(TestConstants.REGISTRY_IDS.value))

        # Test registry entries
        for id, path in zip(
            TestConstants.REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(id, result.registry)
            self.assertEqual(result.registry[id], Path(path))

    def test_missing_file(self):
        # Remove the required file
        self.registry_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestRegistryImporter.run_tests()
