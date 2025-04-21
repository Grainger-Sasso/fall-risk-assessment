from enum import Enum
from pathlib import Path
from typing import Dict, Type

from src.data_io.builders.model_builders.registry.registry_builder import (
    RegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier


class RegistryFileNames(Enum):
    """Enumeration of registry file names."""

    REGISTRY = "registry"


class RegistryImporter(Importer[Registry]):
    """Imports registry data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = RegistryBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path, id_type: Type[Identifier]) -> Registry:
        """Import registry data from directory.

        Args:
            directory (Path): Directory containing registry files

        Returns:
            AggregateFeatureRegistry: Imported registry

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, RegistryFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        registry_file: CSVFile = self.reader.read(
            file_paths[RegistryFileNames.REGISTRY]
        )
        return self.model_builder.build(
            input_file=registry_file, id_type=id_type, subdir_path=directory
        )
