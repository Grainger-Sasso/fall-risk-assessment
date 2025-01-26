from dataclasses import dataclass, field
from typing import List, Dict

from src.data_model.data.imu.imu_metadata import IMUMetadata
from src.data_model.data.imu.sensor.sensor_data import SensorData
from src.data_model.instrument.types.sensor_types import SensorTypes


@dataclass
class IMUData:
    data: List[SensorData]
    metadata: IMUMetadata
    start_time: float
    end_time: float
    _sensor_data_map: Dict[SensorTypes, SensorData] = field(init=False, repr=False)

    def __post_init__(self):
        # Create the axis maps during initialization
        self._sensor_data_map = {
            sensor_data.metadata.sensor_type: sensor_data for sensor_data in self.data
        }

    @property
    def timestamps(self) -> List[float]:
        """
        Returns the start and end time as a list.
        """
        return [self.start_time, self.end_time]

    def _get_data_by_sensor_type(
        self, sensor_data_map: Dict, sensor_type: SensorTypes
    ) -> SensorData:
        """
        Generalized helper function for retrieving data by anatomical or sensor axis.
        """
        if sensor_type not in sensor_data_map:
            raise ValueError(f"{sensor_type} not present in data")
        return sensor_data_map[sensor_type]
