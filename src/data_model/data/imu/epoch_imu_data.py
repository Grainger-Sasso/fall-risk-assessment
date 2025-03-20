from dataclasses import dataclass, field
from typing import Dict, List

from src.data_model.data.imu.sensor_data import SensorData
from src.data_types.instrument.sensor_type import SensorType


@dataclass
class EpochIMUData:
    data: List[SensorData]
    epoch_start_time: float
    epoch_end_time: float
    _sensor_data_map: Dict[SensorType, SensorData] = field(init=False, repr=False)

    def __post_init__(self):
        # Create the axis maps during initialization
        self._sensor_data_map = {
            sensor_data.metadata.sensor_type: sensor_data for sensor_data in self.data
        }

    @property
    def epoch_timestamps(self) -> List[float]:
        """
        Returns the start and end time as a list.
        """
        return [self.epoch_start_time, self.epoch_end_time]

    def get_sensor_data_from_type(self, sensor_type: SensorType):
        return self._sensor_data_map[sensor_type]
