from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.feature_set.feature_set_builder import (
    FeatureSetBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.data_model.feature_set.feature_set import FeatureSet


class FeatureSetFileNames(Enum):
    """Enumeration of feature set file names."""

    FEATURE_SET = "feature_set"


class FeatureSetImporter(Importer[FeatureSet]):
    """Imports feature set data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = FeatureSetBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path, feature_set_name: str) -> FeatureSet:
        """Import feature set data from directory.

        Args:
            directory (Path): Directory containing feature set files
            feature_set_name (str): Name of feature set to import

        Returns:
            Feature set: Imported feature set

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, FeatureSetFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        feature_set_file: CSVFile = self.reader.read(
            file_paths[FeatureSetFileNames.FEATURE_SET]
        )
        return self.model_builder.build(feature_set_file, feature_set_name)
