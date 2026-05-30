import os
from pathlib import Path

from src.data_io.builders.file_builders.instrument_specifications.instrument_specification_file_builder import (
    InstrumentSpecificationFileBuilder,
)
from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationFileNames,
)
from src.data_io.read_write.writers.json.json_dict_file_writer import JSONDictFileWriter
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.identifiers.identifier import Identifier


class InstrumentSpecificationFileExporter(Exporter[IMUSpecifications]):
    sub_dir_name = "instrument_specification_"

    def __init__(self):
        file_builder = InstrumentSpecificationFileBuilder()
        writer = JSONDictFileWriter()
        suffix = "json"
        super().__init__(
            file_builder,
            writer,
            suffix,
            InstrumentSpecificationFileExporter.sub_dir_name,
        )

    def _get_object_id(self, data: IMUSpecifications) -> Identifier:
        return data.specification_id

    def _construct_file_path(self, output_subdir_path: Path) -> Path:
        spec_file_name = (
            f"{InstrumentSpecificationFileNames.INSTRUMENT_SPEC.value}.{self.suffix}"
        )
        return Path(os.path.join(output_subdir_path, spec_file_name))
