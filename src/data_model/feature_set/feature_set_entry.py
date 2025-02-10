from dataclasses import dataclass

from src.identifiers.feature.feature_identifier import FeatureIdentifier


@dataclass
class FeatureSetEntry:
    feature_identifier: FeatureIdentifier
