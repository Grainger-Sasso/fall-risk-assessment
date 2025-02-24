from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.dataset.dataset_fields import DatasetFields
from src.data_model.dataset.dataset import Dataset
from src.data_model.dataset.dataset_entry import DatasetEntry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class DatasetBuilder(ModelBuilder):
    """Builds Dataset from input CSV file"""

    version = "1.0"

    def build(self, input_file: CSVFile, dataset_name: str) -> Dataset:
        """Build Dataset from CSV file.

        Args:
            input_file (CSVFile): Input CSV file containing dataset entries
            dataset_name (str): Name of the dataset

        Returns:
            Dataset: Built dataset object

        Raises:
            ValueError: If input file is invalid
        """
        if not isinstance(input_file, CSVFile) or not dataset_name:
            raise ValueError("Invalid dataset CSV file or name")

        user_data_identifiers = [
            UserIdentifier(id)
            for id in input_file.data[DatasetFields.USER_DATA_IDENTIFIER.value]
        ]
        imu_data_identifiers = [
            IMUDataIdentifier(id)
            for id in input_file.data[DatasetFields.IMU_DATA_IDENTIFIER.value]
        ]
        entries = [
            DatasetEntry(user_data_id=user_id, imu_data_id=imu_id)
            for user_id, imu_id in zip(user_data_identifiers, imu_data_identifiers)
        ]
        return Dataset(name=dataset_name, entries=entries)
