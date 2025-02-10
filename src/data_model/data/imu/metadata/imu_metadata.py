from dataclasses import dataclass
from typing import Dict

from src.data_model.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.data_model.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis


@dataclass
class IMUMetadata:
    imu_data_identifier: IMUDataIdentifier
    instument_identifier: InstrumentIdentifier
    imu_orientation_map: Dict[SensorAxis:AnatomicalAxis]
