from typing import List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.dataset.dataset_fields import DatasetFields
from src.data_model.dataset.dataset import Dataset
from src.data_model.dataset.dataset_entry import DatasetEntry
from src.identifiers.user.user_identifier import UserIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class DatasetBuilder(ModelBuilder):
    """Builds Dataset from input CSV file"""

    version = "1.0"

    def build(self, input_file: CSVFile, dataset_name: str) -> Dataset:
        user_data_identifiers = [
            UserIdentifier(id)
            for id in input_file.data[DatasetFields.USER_DATA_IDENTIFIER]
        ]
        imu_data_identifiers = [
            IMUDataIdentifier(id)
            for id in input_file.data[DatasetFields.IMU_DATA_IDENTIFIER]
        ]
        entries: List[DatasetEntry] = []
        for user_data_id, imu_data_id in zip(
            user_data_identifiers, imu_data_identifiers
        ):
            entries.append(
                DatasetEntry(user_data_id=user_data_id, imu_data_id=imu_data_id)
            )
        return Dataset(name=dataset_name, entries=entries)
