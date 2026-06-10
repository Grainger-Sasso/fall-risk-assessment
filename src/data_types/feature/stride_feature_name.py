from typing import List, Union

from src.data_types.feature.feature_type import FeatureType
from src.data_types.feature.mobgap_feature_type import MobgapFeatureType

StrideFeatureName = Union[FeatureType, MobgapFeatureType]


def parse_stride_feature_name(value: str) -> StrideFeatureName:
    if value.startswith("mobgap:"):
        return MobgapFeatureType(value)
    return FeatureType(value)


def stride_feature_name_value(name: StrideFeatureName) -> str:
    return name.value


def is_mobgap_stride_name(name: StrideFeatureName) -> bool:
    return isinstance(name, MobgapFeatureType)
