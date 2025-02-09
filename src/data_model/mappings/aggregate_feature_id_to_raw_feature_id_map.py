from dataclasses import dataclass
from typing import Dict

from src.data_model.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.data_model.identifiers.feature.raw_feature_identifier import (
    RawFeatureIdentifier,
)


@dataclass
class AggregateFeatureIDToRawFeatureIDMap:
    map: Dict[AggregateFeatureIdentifier:RawFeatureIdentifier]
