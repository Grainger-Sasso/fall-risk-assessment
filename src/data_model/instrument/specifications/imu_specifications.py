from typing import List
from dataclasses import dataclass

from src.data_model.instrument.types.sensor_types import SensorTypes
from src.data_model.instrument.specifications.sensor_specifications import (
    SensorSpecifications,
)


@dataclass
class IMUSpecifications:
    # TODO: properly implement hashable id
    specification_id: str
    imu_name: str
    sensor_types = List[SensorTypes]
    sensor_specifications: dict[SensorTypes:SensorSpecifications]
