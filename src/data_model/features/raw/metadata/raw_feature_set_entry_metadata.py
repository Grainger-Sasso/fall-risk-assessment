from dataclasses import dataclass

from src.identifiers.feature.raw_feature_identifier import (
    RawFeatureIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class RawFeatureSetEntryMetadata:
    raw_feature_identifier: RawFeatureIdentifier
    user_identifier: UserIdentifier
    imu_data_identifier: IMUDataIdentifier
    start_time: float
    epoch_length: float
