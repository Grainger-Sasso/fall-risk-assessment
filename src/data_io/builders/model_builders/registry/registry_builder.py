from pathlib import Path
from typing import Dict, Type

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.registry.registry_fields import RegistryFields
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier


class RegistryBuilder(ModelBuilder):
    version = "1.0"

    def build(self, input_file: CSVFile, id_type: Type[Identifier]) -> Registry:
        if not isinstance(input_file, CSVFile):
            raise ValueError("Invalid registry CSV file")
        return self.build_registry(input_file, id_type)

    def build_registry(self, input_file: CSVFile, id_type: Type[Identifier]) -> Dict[str, Path]:
        registry: Dict[str, Path] = {}
        ids = input_file.data[RegistryFields.DATA_IDENTIFIER.value]
        paths = input_file.data[RegistryFields.DIRECTORY.value]
        for id, path in zip(ids, paths):
            registry[id] = Path(path)
        return Registry(registry, id_type)
