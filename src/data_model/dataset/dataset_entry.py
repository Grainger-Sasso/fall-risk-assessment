from dataclasses import dataclass

from src.identifiers.user.user_identifier import UserIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class DatasetEntry:
    user_data_id: UserIdentifier
    imu_data_id: IMUDataIdentifier
