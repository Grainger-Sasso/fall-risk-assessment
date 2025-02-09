from dataclasses import dataclass

from src.data_model.features.types.raw_feature_type import RawFeatureType


@dataclass
class RawFeature:
    """
    Represents raw feature
    """

    feature_type: RawFeatureType
    value: float
