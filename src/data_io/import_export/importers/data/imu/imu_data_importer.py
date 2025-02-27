from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_model.data.imu.imu_data import IMUData


class IMUDataFileNames(Enum):
    IMU_DATA = "imu_data"


class IMUDataImporter(Importer[IMUData]):
    def __init__(self):
        reader = HDF5FileReader()
        model_builder = IMUDataBuilder()
        file_suffixes = ["h5"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> IMUData:
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, IMUDataFileNames
        )
        imu_data_group: HDF5Group = self.reader.read(
            file_paths[IMUDataFileNames.IMU_DATA]
        )
        return self.model_builder.build(imu_data_group)
