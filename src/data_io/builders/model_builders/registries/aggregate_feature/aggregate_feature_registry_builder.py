from typing import Dict
from pathlib import Path
from src.data_io.builders.model_builders.registries.registry_builder import (
    RegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.database_manager.registries.aggregate_feature.aggregate_feature_registry import (
    AggregateFeatureRegistry,
)


class AggregateFeatureRegistryBuilder(RegistryBuilder):
    version = "1.0"
    
    def build(self, input_file: CSVFile) -> AggregateFeatureRegistry:
        registry: Dict[str, Path] = self.build_registry(input_file)
        return AggregateFeatureRegistry(
            {AggregateFeatureIdentifier(id): path for id, path in registry.items()}
        )
