from dataclasses import dataclass

from src.identifiers.feature.feature_identifier import (
    FeatureIdentifier,
)
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


@dataclass
class FeatureMetadata:
    feature_identifier: FeatureIdentifier
    user_identifier: UserIdentifier
    imu_data_identifier: IMUDataIdentifier
