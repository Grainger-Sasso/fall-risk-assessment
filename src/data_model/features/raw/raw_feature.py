from dataclasses import dataclass

from data_types.feature.raw_feature_type import RawFeatureType


@dataclass
class RawFeature:
    """
    Represents raw feature
    """

    feature_type: RawFeatureType
    value: float
