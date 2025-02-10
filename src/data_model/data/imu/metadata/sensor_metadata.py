from dataclasses import dataclass

from src.data_types.instrument.sensor_type import SensorType
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.units.unit import Unit


@dataclass
class SensorMetadata:
    """
    Metadata for SensorData
    """

    sensor_type: SensorType
    sampling_rate: float
    sensor_orientation_map: dict[SensorAxis:AnatomicalAxis]
    unit: Unit
