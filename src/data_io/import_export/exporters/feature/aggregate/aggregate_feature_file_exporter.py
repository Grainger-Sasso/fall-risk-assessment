import os
from pathlib import Path

from src.data_io.builders.file_builders.feature.aggregate.aggregate_feature_file_builder import (
    AggregateFeatureSetEntryFileBuilder,
)
from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.importers.features.aggregate.aggregate_feature_importer import (
    AggregateFeatureFileNames,
)
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.identifiers.identifier import Identifier


class AggregateFeatureFileExporter(Exporter[AggregateFeatureSetEntry]):
    sub_dir_name = "aggregate_features_"

    def __init__(self):
        file_builder = AggregateFeatureSetEntryFileBuilder()
        writer = HDF5FileWriter()
        suffix = "h5"
        super().__init__(
            file_builder, writer, suffix, AggregateFeatureFileExporter.sub_dir_name
        )

    def _get_object_id(self, data: AggregateFeatureSetEntry) -> Identifier:
        return data.metadata.aggregate_feature_identifier

    def _construct_file_path(self, output_subdir_path: Path) -> Path:
        agg_feature_file_name = (
            f"{AggregateFeatureFileNames.AGGREGATE_FEATURES.value}.{self.suffix}"
        )
        return Path(os.path.join(output_subdir_path, agg_feature_file_name))
