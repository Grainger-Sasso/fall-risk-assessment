from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.registries.imu.imu_data_registry_builder import (
    IMUDataRegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.database_manager.registries.imu.imu_data_registry import IMUDataRegistry


class IMUDataRegistryFileNames(Enum):
    """Enumeration of registry file names."""

    IMU_DATA_REGISTRY = "imu_data_registry"


class IMUDataRegistryImporter(Importer[IMUDataRegistry]):
    """Imports registry data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = IMUDataRegistryBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> IMUDataRegistry:
        """Import registry data from directory.

        Args:
            directory (Path): Directory containing registry files

        Returns:
            IMUDataRegistry: Imported registry

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, IMUDataRegistryFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise FileNotFoundError(
                f"Files missing in set of file paths found: {file_paths}"
            )

        registry_file: CSVFile = self.reader.read(
            file_paths[IMUDataRegistryFileNames.IMU_DATA_REGISTRY]
        )
        return self.model_builder.build(registry_file) 