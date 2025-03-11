from pathlib import Path

from src.data_io.builders.file_builders.mapping.mapping_file_builder import (
    MappingFileBuilder,
)
from src.data_io.import_export.importers.mapping.mapping_importer import (
    MappingFileNames,
)
from src.data_io.read_write.writers.csv.csv_file_writer import CSVFileWriter
from src.database_manager.mapping.mapping import Mapping


class MappingExporter:
    def __init__(self):
        self.file_builder = MappingFileBuilder()
        self.writer = CSVFileWriter()



