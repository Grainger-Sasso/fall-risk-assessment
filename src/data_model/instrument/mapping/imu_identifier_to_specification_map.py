from dataclasses import dataclass
from typing import Dict

from src.data_model.instrument.identifiers.imu_identifier import IMUIdentifier
from src.data_model.instrument.specifications.imu_specifications import (
    IMUSpecifications,
)


@dataclass
class IMUIdentifierToSpecificationMap:
    map: Dict[IMUIdentifier:IMUSpecifications]
