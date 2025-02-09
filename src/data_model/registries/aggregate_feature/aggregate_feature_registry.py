from pathlib import Path
from typing import Dict

from src.data_model.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.data_model.registries.registry import Registry


class AggregateFeatureRegistry(Registry):
    def __init__(self, registry: Dict[AggregateFeatureIdentifier:Path]):
        super().__init__(registry)
