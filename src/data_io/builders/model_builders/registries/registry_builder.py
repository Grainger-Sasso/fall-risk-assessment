from abc import ABC
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.registries.registry_fields import RegistryFields


class RegistryBuilder(ModelBuilder, ABC):
    version = "1.0"

    def build_registry(self, input_file: CSVFile) -> Dict[str, Path]:
        if not isinstance(input_file, CSVFile):
            raise ValueError("Input must be an HDF5Group")
        registry: Dict[str, Path] = {}
        ids = input_file.data[RegistryFields.DATA_IDENTIFIER.value]
        paths = input_file.data[RegistryFields.DIRECTORY.value]
        for id, path in zip(ids, paths):
            registry[id] = Path(path)
        return registry
