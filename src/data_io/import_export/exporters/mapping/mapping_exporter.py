from src.data_io.builders.file_builders.mapping.mapping_file_builder import (
    MappingFileBuilder,
)
from src.data_io.import_export.exporters.database_exporter import DatabaseExporter
from src.data_io.import_export.importers.mapping.mapping_importer import (
    MappingFileNames,
)
from src.data_io.read_write.writers.csv.csv_file_writer import CSVFileWriter
from src.database_manager.mapping.mapping import Mapping


class MappingExporter(DatabaseExporter[Mapping]):
    def __init__(self):
        file_builder = MappingFileBuilder()
        writer = CSVFileWriter()
        suffix: str = "csv"
        super().__init__(file_builder, writer, suffix)

    def _get_file_name(self) -> str:
        return MappingFileNames.MAPPING.value
