import os
from pathlib import Path

from src.data_io.builders.file_builders.feature.raw.raw_feature_file_builder import (
    RawFeatureSetEntryFileBuilder,
)
from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.importers.features.raw.raw_feature_importer import (
    RawFeatureFileNames,
)
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.identifiers.identifier import Identifier


class RawFeatureFileExporter(Exporter[RawFeatureSetEntry]):
    sub_dir_name = "raw_features_"

    def __init__(self):
        file_builder = RawFeatureSetEntryFileBuilder()
        writer = HDF5FileWriter()
        suffix = "h5"
        super().__init__(
            file_builder, writer, suffix, RawFeatureFileExporter.sub_dir_name
        )

    def _get_object_id(self, data: RawFeatureSetEntry) -> Identifier:
        return data.metadata.raw_feature_identifier

    def _construct_file_path(self, output_subdir_path: Path) -> Path:
        raw_feature_file_name = (
            f"{RawFeatureFileNames.RAW_FEATURES.value}.{self.suffix}"
        )
        return Path(os.path.join(output_subdir_path, raw_feature_file_name))
