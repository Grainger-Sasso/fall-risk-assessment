from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.registries.registry_builder import (
    RegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.database_manager.registries.raw_feature.raw_feature_registry import (
    RawFeatureRegistry,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier


class RawFeatureRegistryBuilder(RegistryBuilder):
    version = "1.0"

    def build(self, input_file: CSVFile) -> RawFeatureRegistry:
        registry: Dict[str, Path] = self.build_registry(input_file)
        return RawFeatureRegistry(
            {RawFeatureIdentifier(id): path for id, path in registry.items()}
        )
