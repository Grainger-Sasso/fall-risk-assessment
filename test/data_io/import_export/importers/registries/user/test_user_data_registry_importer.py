import tempfile
from pathlib import Path

from src.data_io.import_export.importers.registries.user.user_data_registry_importer import (
    UserDataRegistryFileNames,
    UserDataRegistryImporter,
)
from src.database_manager.registries.user.user_data_registry import UserDataRegistry
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestUserDataRegistryImporter(BaseTest):
    def setUp(self):
        self.importer = UserDataRegistryImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = RegistryHelper()

        # Create test file in temp directory
        self.registry_path = self.helper.create_test_registry_file(
            TestConstants.USER_REGISTRY_IDS.value,
            self.temp_path / f"{UserDataRegistryFileNames.USER_DATA_REGISTRY.value}.csv",
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
        self.assertIsInstance(result, UserDataRegistry)
        self.assertEqual(
            len(result.registry), len(TestConstants.USER_REGISTRY_IDS.value)
        )

        # Test registry entries
        for id, path in zip(
            TestConstants.USER_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(UserIdentifier(id), result.registry)
            self.assertEqual(result.registry[UserIdentifier(id)], Path(path))

    def test_missing_file(self):
        # Remove the required file
        self.registry_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestUserDataRegistryImporter.run_tests() 