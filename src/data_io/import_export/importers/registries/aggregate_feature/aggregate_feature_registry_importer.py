from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.registries.aggregate_feature.aggregate_feature_registry_builder import (
    AggregateFeatureRegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.database_manager.registries.aggregate_feature.aggregate_feature_registry import (
    AggregateFeatureRegistry,
)


class AggregateFeatureRegistryFileNames(Enum):
    """Enumeration of registry file names."""

    AGGREGATE_FEATURE_REGISTRY = "aggregate_feature_registry"


class AggregateFeatureRegistryImporter(Importer[AggregateFeatureRegistry]):
    """Imports registry data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = AggregateFeatureRegistryBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> AggregateFeatureRegistry:
        """Import registry data from directory.

        Args:
            directory (Path): Directory containing registry files

        Returns:
            AggregateFeatureRegistry: Imported registry

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, AggregateFeatureRegistryFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        registry_file: CSVFile = self.reader.read(
            file_paths[AggregateFeatureRegistryFileNames.AGGREGATE_FEATURE_REGISTRY]
        )
        return self.model_builder.build(registry_file)
