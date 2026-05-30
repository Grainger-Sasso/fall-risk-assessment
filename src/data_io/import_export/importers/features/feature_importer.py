from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.features.record_feature_builder import (
    RecordFeatureBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_model.features.record_features import RecordFeatures


class FeatureFileNames(Enum):
    """Enumeration of feature file names."""

    FEATURES = "features"


class FeatureImporter(Importer[RecordFeatures]):
    """Imports record-level feature data from files."""

    def __init__(self):
        reader = HDF5FileReader()
        model_builder = RecordFeatureBuilder()
        file_suffixes = ["h5"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> RecordFeatures:
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, FeatureFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        feature_file: HDF5Group = self.reader.read(file_paths[FeatureFileNames.FEATURES])
        return self.model_builder.build(feature_file)
