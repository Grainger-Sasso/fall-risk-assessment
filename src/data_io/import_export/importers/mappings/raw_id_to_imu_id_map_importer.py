from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.mappings.raw_id_to_imu_id_map_builder import (
    RawFeatureIDToIMUIDMapBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.database_manager.mappings.raw_feature_id_to_imu_data_id_map import (
    RawFeatureIDToIMUDataIDMap,
)


class RawIDToIMUIDMapFileNames(Enum):
    """Enumeration of mapping file names."""

    RAW_FEATURE_TO_IMU_DATA_ID_MAP = "raw_feature_to_imu_data_id_map"


class RawIDToIMUIDMapImporter(Importer[RawFeatureIDToIMUDataIDMap]):
    """Imports map data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = RawFeatureIDToIMUIDMapBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> RawFeatureIDToIMUDataIDMap:
        """Import map data from directory.

        Args:
            directory (Path): Directory containing map files

        Returns:
            RawFeatureIDToIMUDataIDMap: Imported map

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, RawIDToIMUIDMapFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise FileNotFoundError(
                f"Files missing in set of file paths found: {file_paths}"
            )

        map_file: CSVFile = self.reader.read(
            file_paths[RawIDToIMUIDMapFileNames.RAW_FEATURE_TO_IMU_DATA_ID_MAP]
        )
        return self.model_builder.build(map_file)
