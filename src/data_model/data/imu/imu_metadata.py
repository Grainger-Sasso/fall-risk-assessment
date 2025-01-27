from dataclasses import dataclass

from src.data_model.instrument.identifiers.imu_identifier import IMUIdentifier
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis


@dataclass
class IMUMetadata:
    imu_identifier: IMUIdentifier
    imu_orientation_map: dict[SensorAxis:AnatomicalAxis]
