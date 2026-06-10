from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.user.user_data import UserData
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.processing.backend_requirements import validate_extraction_requirements
from src.gait_features.processing.imu_validation import ImuValidationMixin

try:
    from mobgap.data import GaitDatasetFromData
    from mobgap.utils.conversions import to_body_frame
except ModuleNotFoundError:
    GaitDatasetFromData = None
    to_body_frame = None


class MobgapImuAdapter(ImuValidationMixin):
    def build_datapoint(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications] = None,
    ) -> Tuple[Any, SensorData, float]:
        if GaitDatasetFromData is None or to_body_frame is None:
            raise ModuleNotFoundError(
                "mobgap is required for MobGap gait feature extraction."
            )
        validate_extraction_requirements(
            imu_data=imu_data,
            backend=GaitExtractionBackendId.MOBGAP,
        )
        sensor_data, accel, _ = self.prepare_accelerometer_for_backend(
            imu_data=imu_data,
            instrument_specifications=instrument_specifications,
            treadmill_profile=False,
            backend=GaitExtractionBackendId.MOBGAP,
        )
        _, gyro = self.prepare_gyroscope_for_backend(
            imu_data=imu_data,
            backend=GaitExtractionBackendId.MOBGAP,
        )
        sampling_rate_hz = float(sensor_data.metadata.sampling_rate)
        frame = self._build_sensor_frame(accel=accel, gyro=gyro)
        frame = to_body_frame(frame)
        height_m = self.normalize_height(user_data.clinical_demographic_data.height.value)
        participant_metadata = {
            "height_m": height_m,
            "sensor_height_m": self.default_sensor_height_m(height_m),
        }
        datapoint = GaitDatasetFromData(
            data={("participant",): {"LowerBack": frame}},
            sampling_rate_hz=sampling_rate_hz,
            participant_metadata=participant_metadata,
            single_sensor_name="LowerBack",
        )
        return datapoint[0], sensor_data, sampling_rate_hz

    @staticmethod
    def _build_sensor_frame(accel: np.ndarray, gyro: np.ndarray) -> pd.DataFrame:
        if gyro.shape[0] != accel.shape[0]:
            raise ValueError(
                "Accelerometer and gyroscope streams must have equal sample counts "
                f"({accel.shape[0]} vs {gyro.shape[0]})."
            )
        return pd.DataFrame(
            {
                "acc_x": accel[:, 0],
                "acc_y": accel[:, 1],
                "acc_z": accel[:, 2],
                "gyr_x": gyro[:, 0],
                "gyr_y": gyro[:, 1],
                "gyr_z": gyro[:, 2],
            }
        )
