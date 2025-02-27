from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.registries.registry_builder import (
    RegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.database_manager.registries.user.user_data_registry import UserDataRegistry
from src.identifiers.user.user_identifier import UserIdentifier


class UserDataRegistryBuilder(RegistryBuilder):
    version = "1.0"

    def build(self, input_file: CSVFile) -> UserDataRegistry:
        registry: Dict[str, Path] = self.build_registry(input_file)
        return UserDataRegistry(
            {UserIdentifier(id): path for id, path in registry.items()}
        )
