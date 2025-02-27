import tempfile
from pathlib import Path

from src.data_io.import_export.importers.dataset.dataset_importer import (
    DatasetFileNames,
    DatasetImporter,
)
from src.data_model.dataset.dataset import Dataset
from src.data_model.dataset.dataset_entry import DatasetEntry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import DatasetHelper, TestConstants


class TestDatasetImporter(BaseTest):
    def setUp(self):
        self.importer = DatasetImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = DatasetHelper()

        # Create test file in temp directory
        self.dataset_path = self.helper.create_test_dataset_file(
            self.temp_path / f"{DatasetFileNames.DATASET.value}.csv"
        )

    def tearDown(self):
        # Clean up test files
        if self.dataset_path.exists():
            self.dataset_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(
            self.temp_path, TestConstants.DATASET_NAME.value
        )

        # Assertions
        self.assertIsInstance(result, Dataset)
        self.assertEqual(result.name, TestConstants.DATASET_NAME.value)

        # Test entries
        self.assertIsInstance(result.entries, list)
        self.assertEqual(len(result.entries), len(TestConstants.DATASET_USER_IDS.value))

        for ix, entry in enumerate(result.entries):
            self.assertIsInstance(entry, DatasetEntry)
            self.assertIsInstance(entry.user_data_id, UserIdentifier)
            self.assertIsInstance(entry.imu_data_id, IMUDataIdentifier)
            self.assertEqual(
                entry.user_data_id.value, TestConstants.DATASET_USER_IDS.value[ix]
            )
            self.assertEqual(
                entry.imu_data_id.value, TestConstants.DATASET_IMU_IDS.value[ix]
            )

    def test_missing_file(self):
        # Remove the required file
        self.dataset_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path, "")


if __name__ == "__main__":
    TestDatasetImporter.run_tests()
