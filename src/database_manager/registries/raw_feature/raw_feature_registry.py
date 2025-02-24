from pathlib import Path
from typing import Dict

from src.identifiers.feature.raw_feature_identifier import (
    RawFeatureIdentifier,
)
from src.database_manager.registries.registry import Registry


class RawFeatureRegistry(Registry):
    def __init__(self, registry: Dict[RawFeatureIdentifier, Path]):
        super().__init__(registry)
