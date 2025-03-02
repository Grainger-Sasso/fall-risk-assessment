from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.registries.raw_feature.raw_feature_registry_builder import (
    RawFeatureRegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.database_manager.registries.raw_feature.raw_feature_registry import (
    RawFeatureRegistry,
)


class RawFeatureRegistryFileNames(Enum):
    """Enumeration of registry file names."""

    RAW_FEATURE_REGISTRY = "raw_feature_registry"


class RawFeatureRegistryImporter(Importer[RawFeatureRegistry]):
    """Imports registry data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = RawFeatureRegistryBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> RawFeatureRegistry:
        """Import registry data from directory.

        Args:
            directory (Path): Directory containing registry files

        Returns:
            RawFeatureRegistry: Imported registry

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, RawFeatureRegistryFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise FileNotFoundError(
                f"Files missing in set of file paths found: {file_paths}"
            )

        registry_file: CSVFile = self.reader.read(
            file_paths[RawFeatureRegistryFileNames.RAW_FEATURE_REGISTRY]
        )
        return self.model_builder.build(registry_file) 