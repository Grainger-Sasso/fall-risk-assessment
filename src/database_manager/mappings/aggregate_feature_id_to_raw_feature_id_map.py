from typing import Dict

from src.database_manager.mappings.mapping import Mapping
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier


class AggregateFeatureIDToRawFeatureIDMap(
    Mapping[AggregateFeatureIdentifier, RawFeatureIdentifier]
):
    """Maps aggregate feature IDs to raw feature IDs"""

    def __init__(self, map: Dict[AggregateFeatureIdentifier, RawFeatureIdentifier]):
        super().__init__(map)
