from abc import ABC
from typing import Dict
from pathlib import Path
from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.registries.registry_fields import RegistryFields


class RegistryBuilder(ModelBuilder, ABC):
    def build_registry(self, input_file: CSVFile) -> Dict[str, Path]:
        registry: Dict[str, Path] = {}
        ids = input_file[RegistryFields.DATA_IDENTIFIER]
        paths = input_file[RegistryFields.DIRECTORY]
        for id, path in zip(ids, paths):
            registry[id] = Path(path)
        return registry
