from typing import Dict

from src.database_manager.mappings.mapping import Mapping
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class RawFeatureIDToIMUDataIDMap(Mapping[RawFeatureIdentifier, IMUDataIdentifier]):
    """Maps raw feature IDs to IMU data IDs"""

    def __init__(self, map: Dict[RawFeatureIdentifier, IMUDataIdentifier]):
        super().__init__(map)
