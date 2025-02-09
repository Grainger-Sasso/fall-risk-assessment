from dataclasses import dataclass
from typing import Dict

from src.data_model.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.data_model.identifiers.user.user_identifier import UserIdentifier


@dataclass
class IMUDataIDToUserDataIDMap:
    map: Dict[IMUDataIdentifier:UserIdentifier]
