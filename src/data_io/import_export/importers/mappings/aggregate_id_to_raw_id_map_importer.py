from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.mappings.aggregate_id_to_raw_id_map_builder import (
    AggregateIDToRawIDMapBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.database_manager.mappings.aggregate_feature_id_to_raw_feature_id_map import (
    AggregateFeatureIDToRawFeatureIDMap,
)


class AggregateIDToRawIDMapFileNames(Enum):
    """Enumeration of mapping file names."""

    AGGREGATE_FEATURE_TO_RAW_FEATURE_ID_MAP = "aggregate_feature_to_raw_feature_id_map"


class AggregateIDToRawIDMapImporter(Importer[AggregateFeatureIDToRawFeatureIDMap]):
    """Imports map data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = AggregateIDToRawIDMapBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> AggregateFeatureIDToRawFeatureIDMap:
        """Import map data from directory.

        Args:
            directory (Path): Directory containing map files

        Returns:
            AggregateIDToRawIDMap: Imported map

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, AggregateIDToRawIDMapFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        map_file: CSVFile = self.reader.read(
            file_paths[
                AggregateIDToRawIDMapFileNames.AGGREGATE_FEATURE_TO_RAW_FEATURE_ID_MAP
            ]
        )
        return self.model_builder.build(map_file)
