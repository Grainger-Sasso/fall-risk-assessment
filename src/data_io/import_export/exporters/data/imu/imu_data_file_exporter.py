import os
from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.file_builders.data.imu.imu_data_file_builder import (
    IMUDataFileBuilder,
)
from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataFileNames,
)
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.data.imu.imu_data import IMUData
from src.identifiers.identifier import Identifier


class IMUDataFileExporter(Exporter[IMUData]):
    sub_dir_name = "imu_data_"

    def __init__(self):
        file_builder = IMUDataFileBuilder()
        writer = HDF5FileWriter()
        suffix = "h5"
        super().__init__(file_builder, writer, suffix, IMUDataFileExporter.sub_dir_name)

    def _get_object_id(self, data: IMUData) -> Identifier:
        return data.metadata.imu_data_identifier

    def _construct_file_path(self, subdir_path: Path) -> Path:
        imu_file_name = f"{IMUDataFileNames.IMU_DATA.value}.{self.suffix}"
        return Path(os.path.join(subdir_path, imu_file_name))
