from dataclasses import dataclass
from typing import List

from src.data_model.assessment_data import AssessmentData
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.identifiers.identifier import Identifier


@dataclass
class IMUData(AssessmentData):
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

    def get_data_id(self) -> Identifier:
        return self.metadata.imu_data_identifier
