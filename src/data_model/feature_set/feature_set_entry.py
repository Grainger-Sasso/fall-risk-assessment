from dataclasses import dataclass

from src.data_model.identifiers.feature.feature_identifier import FeatureIdentifier


@dataclass
class FeatureSetEntry:
    feature_identifier: FeatureIdentifier
