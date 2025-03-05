from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.mapping.mapping_builder import MappingBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.database_manager.mapping.mapping import Mapping


class MappingFileNames(Enum):
    """Enumeration of mapping file names."""

    MAPPING = "MAPPING"


class MappingImporter(Importer[Mapping]):
    """Imports map data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = MappingBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> Mapping:
        """Import map data from directory.

        Args:
            directory (Path): Directory containing map file

        Returns:
            Mapping: Imported map

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, MappingFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        map_file: CSVFile = self.reader.read(file_paths[MappingFileNames.MAPPING])
        return self.model_builder.build(map_file)
