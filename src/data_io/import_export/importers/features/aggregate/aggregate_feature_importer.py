from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.features.aggregate.aggregate_feature_set_entry_builder import (
    AggregateFeatureSetEntryBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)


class AggregateFeatureFileNames(Enum):
    """Enumeration of aggregate feature file names."""

    AGGREGATE_FEATURES = "aggregate_features"


class AggregateFeatureImporter(Importer[AggregateFeatureSetEntry]):
    """Imports aggregate feature data from files."""

    def __init__(self):
        reader = HDF5FileReader()
        model_builder = AggregateFeatureSetEntryBuilder()
        file_suffixes = ["h5"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> AggregateFeatureSetEntry:
        """Import aggregate feature data from directory.

        Args:
            directory (Path): Directory containing aggregate feature files

        Returns:
            AggregateFeatureSetEntry: Imported aggregate feature set entry

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, AggregateFeatureFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        aggregate_feature_file: HDF5Group = self.reader.read(
            file_paths[AggregateFeatureFileNames.AGGREGATE_FEATURES]
        )
        return self.model_builder.build(aggregate_feature_file)
