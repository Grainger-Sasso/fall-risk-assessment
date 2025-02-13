from dataclasses import dataclass

from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)


@dataclass
class FeatureSetEntry:
    raw_feature_identifier: RawFeatureIdentifier
    aggregate_feature_identifier: AggregateFeatureIdentifier
