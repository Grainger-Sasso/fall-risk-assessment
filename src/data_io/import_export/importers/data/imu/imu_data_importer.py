from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata


class IMUDataFileNames(Enum):
    IMU_DATA = "imu_data"


class IMUDataImporter(Importer[IMUData]):
    def __init__(self):
        reader = HDF5FileReader()
        model_builder = IMUDataBuilder()
        file_suffixes = ["h5"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> IMUData:
        imu_data_group: HDF5Group = self.reader.read(self._resolve_imu_file(directory))
        return self.model_builder.build(imu_data_group)

    def import_header(self, directory: Path) -> IMUMetadata:
        """Import only the IMU metadata, without loading any sensor arrays."""
        attributes = self.reader.read_top_group_attributes(self._resolve_imu_file(directory))
        return self.model_builder.build_metadata(attributes)

    def _resolve_imu_file(self, directory: Path) -> Path:
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, IMUDataFileNames
        )
        if not all(p.exists() for p in file_paths.values()):
            raise FileNotFoundError(
                f"Files missing in set of file paths found: {file_paths}"
            )
        return file_paths[IMUDataFileNames.IMU_DATA]
