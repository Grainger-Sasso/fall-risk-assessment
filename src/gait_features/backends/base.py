from typing import Optional, Protocol

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.contracts.extraction_result import GaitExtractionResult


class GaitExtractionBackend(Protocol):
    def extract(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications],
        profile: ExtractionProfile,
    ) -> GaitExtractionResult: ...
