from dataclasses import dataclass
from typing import List

from src.data_model.data.imu.imu_metadata import IMUMetadata
from src.data_model.data.imu.epoch_imu_data import EpochIMUData


@dataclass
class IMUData:
    """
    Represents IMU recordings
    """
    data: List[EpochIMUData]
    metadata: IMUMetadata
    start_time: float
    end_time: float

    @property
    def timestamps(self) -> List[float]:
        """
        Returns the start and end time as a list.
        """
        return [self.start_time, self.end_time]
