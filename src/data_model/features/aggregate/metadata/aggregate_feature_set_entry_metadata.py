from dataclasses import dataclass

from src.data_model.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.data_model.identifiers.feature.raw_feature_identifier import (
    RawFeatureIdentifier,
)
from src.data_model.identifiers.user.user_identifier import UserIdentifier
from src.data_model.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class AggregateFeatureSetEntryMetadata:
    aggregate_feature_identifier: AggregateFeatureIdentifier
    raw_feature_identifier: RawFeatureIdentifier
    user_identifier: UserIdentifier
    imu_data_identifier: IMUDataIdentifier
