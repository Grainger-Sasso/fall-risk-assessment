from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.features.raw.raw_feature_set_entry_builder import (
    RawFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry


class RawFeatureFileNames(Enum):
    """Enumeration of raw feature file names."""

    RAW_FEATURES = "raw_features"


class RawFeatureImporter(Importer[RawFeatureSetEntryBuilder]):
    """Imports raw feature data from files."""

    def __init__(self):
        reader = HDF5FileReader()
        model_builder = RawFeatureSetEntryBuilder()
        file_suffixes = ["h5"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> RawFeatureSetEntry:
        """Import raw feature data from directory.

        Args:
            directory (Path): Directory containing raw feature files

        Returns:
            RawFeatureSetEntry: Imported raw feature set entry

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, RawFeatureFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        raw_feature_file: HDF5Group = self.reader.read(
            file_paths[RawFeatureFileNames.RAW_FEATURES]
        )
        return self.model_builder.build(raw_feature_file)
