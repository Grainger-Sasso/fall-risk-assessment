import os
from pathlib import Path

from src.data_io.builders.file_builders.feature.record_feature_file_builder import (
    RecordFeatureFileBuilder,
)
from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.importers.features.feature_importer import FeatureFileNames
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.features.record_features import RecordFeatures
from src.identifiers.identifier import Identifier


class FeatureFileExporter(Exporter[RecordFeatures]):
    sub_dir_name = "features_"

    def __init__(self):
        file_builder = RecordFeatureFileBuilder()
        writer = HDF5FileWriter()
        suffix = "h5"
        super().__init__(file_builder, writer, suffix, FeatureFileExporter.sub_dir_name)

    def _get_object_id(self, data: RecordFeatures) -> Identifier:
        return data.feature_metadata.feature_identifier

    def _construct_file_path(self, output_subdir_path: Path) -> Path:
        feature_file_name = f"{FeatureFileNames.FEATURES.value}.{self.suffix}"
        return Path(os.path.join(output_subdir_path, feature_file_name))
