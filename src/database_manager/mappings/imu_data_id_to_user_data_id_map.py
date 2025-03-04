from typing import Dict

from src.database_manager.mappings.mapping import Mapping
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class IMUDataIDToUserDataIDMap(Mapping[IMUDataIdentifier, UserIdentifier]):
    """Maps IMU data IDs to user IDs"""

    def __init__(self, map: Dict[IMUDataIdentifier, UserIdentifier]):
        super().__init__(map)
