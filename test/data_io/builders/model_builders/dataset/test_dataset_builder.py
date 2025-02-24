from src.data_io.builders.model_builders.dataset.dataset_builder import DatasetBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_model.dataset.dataset import Dataset
from src.data_model.dataset.dataset_entry import DatasetEntry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import DatasetHelper, TestConstants


class TestDatasetBuilder(BaseTest):
    def setUp(self):
        self.builder = DatasetBuilder()
        self.data_helper = DatasetHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_dataset_csv()

        # Test building dataset
        result = self.builder.build(csv_data, TestConstants.DATASET_NAME.value)

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

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None, None)


if __name__ == "__main__":
    TestDatasetBuilder.run_tests()
