from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.registry.registry_fields import RegistryFields
from src.database_manager.registry.registry import Registry


class RegistryBuilder(ModelBuilder):
    version = "1.0"

    def build(self, input_file: CSVFile) -> Registry:
        if not isinstance(input_file, CSVFile):
            raise ValueError("Invalid registry CSV file")
        return self.build_registry(input_file)

    def build_registry(self, input_file: CSVFile) -> Dict[str, Path]:
        registry: Dict[str, Path] = {}
        ids = input_file.data[RegistryFields.DATA_IDENTIFIER.value]
        paths = input_file.data[RegistryFields.DIRECTORY.value]
        for id, path in zip(ids, paths):
            registry[id] = Path(path)
        return Registry(registry)
