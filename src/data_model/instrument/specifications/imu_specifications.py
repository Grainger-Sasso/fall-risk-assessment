from typing import List
from dataclasses import dataclass

from src.data_model.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from data_model.instrument.types.sensor_type import SensorType
from src.data_model.instrument.specifications.sensor_specifications import (
    SensorSpecifications,
)


@dataclass
class IMUSpecifications:
    """
    Represents the specifications of an IMU device.
    """

    specification_id: InstrumentSpecificationIdentifier
    imu_name: str
    sensor_types = List[SensorType]
    sensor_specifications: dict[SensorType:SensorSpecifications]
