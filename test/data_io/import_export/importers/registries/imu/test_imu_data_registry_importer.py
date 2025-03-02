import tempfile
from pathlib import Path

from src.data_io.import_export.importers.registries.imu.imu_data_registry_importer import (
    IMUDataRegistryFileNames,
    IMUDataRegistryImporter,
)
from src.database_manager.registries.imu.imu_data_registry import IMUDataRegistry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestIMUDataRegistryImporter(BaseTest):
    def setUp(self):
        self.importer = IMUDataRegistryImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = RegistryHelper()

        # Create test file in temp directory
        self.registry_path = self.helper.create_test_registry_file(
            TestConstants.IMU_REGISTRY_IDS.value,
            self.temp_path
            / f"{IMUDataRegistryFileNames.IMU_DATA_REGISTRY.value}.csv",
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
        self.assertIsInstance(result, IMUDataRegistry)
        self.assertEqual(
            len(result.registry), len(TestConstants.IMU_REGISTRY_IDS.value)
        )

        # Test registry entries
        for id, path in zip(
            TestConstants.IMU_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(IMUDataIdentifier(id), result.registry)
            self.assertEqual(
                result.registry[IMUDataIdentifier(id)], Path(path)
            )

    def test_missing_file(self):
        # Remove the required file
        self.registry_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestIMUDataRegistryImporter.run_tests() 