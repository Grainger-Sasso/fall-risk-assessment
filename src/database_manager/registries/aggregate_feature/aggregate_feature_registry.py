from pathlib import Path
from typing import Dict

from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.database_manager.registries.registry import Registry


class AggregateFeatureRegistry(Registry):
    def __init__(self, registry: Dict[AggregateFeatureIdentifier:Path]):
        super().__init__(registry)
