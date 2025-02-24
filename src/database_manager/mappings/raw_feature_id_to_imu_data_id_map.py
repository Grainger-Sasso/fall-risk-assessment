from dataclasses import dataclass
from typing import Dict

from src.identifiers.feature.raw_feature_identifier import (
    RawFeatureIdentifier,
)
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class RawFeatureIDToIMUDataIDMap:
    map: Dict[RawFeatureIdentifier, IMUDataIdentifier]
