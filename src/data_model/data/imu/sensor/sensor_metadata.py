from dataclasses import dataclass

from src.data_model.instrument.types.sensor_types import SensorTypes
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.units.unit import Unit


@dataclass
class SensorMetadata:
    sensor_type: SensorTypes
    # TODO: properly implement hashable data id
    data_id: str
    sampling_rate: float
    sensor_orientation_map: dict[SensorAxis:AnatomicalAxis]
    unit: Unit
