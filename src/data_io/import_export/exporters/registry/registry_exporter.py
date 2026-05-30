from src.data_io.builders.file_builders.registry.registry_file_builder import (
    RegistryFileBuilder,
)
from src.data_io.import_export.exporters.database_exporter import DatabaseExporter
from src.data_io.import_export.importers.registry.registry_importer import (
    RegistryFileNames,
)
from src.data_io.read_write.writers.csv.csv_file_writer import CSVFileWriter
from src.data_model.registry.registry import Registry


class RegistryExporter(DatabaseExporter[Registry]):
    def __init__(self):
        file_builder = RegistryFileBuilder()
        writer = CSVFileWriter()
        suffix: str = "csv"
        super().__init__(file_builder, writer, suffix)

    def _get_file_name(self) -> str:
        return RegistryFileNames.REGISTRY.value
