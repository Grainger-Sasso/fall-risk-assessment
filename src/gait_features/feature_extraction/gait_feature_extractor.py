import inspect
from typing import Any, Dict

import numpy as np
import skdh

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.user.user_data import UserData
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)

class GaitResults:
    def __init__(self, results: Dict[str, Any]):
        self.data: Dict[str, Any] = results


class GaitFeatureExtractor:
    GRAVITY_M_PER_S2 = 9.80665

    def __init__(self, treadmill_profile: bool = False):
        self.treadmill_profile = treadmill_profile
        self.pipeline: skdh.Pipeline = self._build_pipeline()
        self.gait_res_key = "GaitLumbar"

    def extract_gait_features(self, imu_data: IMUData, user_data: UserData) -> Dict[str, Any]:
        sensor_data: SensorData = imu_data.data[0].data[0]
        time = sensor_data.time
        # Format is expected to be np array of shape (N, 3) where 3 is three axes: X, Y, Z
        accel = np.column_stack(
            (
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data,
            )
        )
        numeric_time = self._coerce_time_for_validation(time)
        self._validate_time(numeric_time=numeric_time, accel=accel)
        accel = self._normalize_accel_units(accel)
        self._validate_sensor_sanity(accel=accel)
        height = self._normalize_height(user_data.clinical_demographic_data.height.value)
        self._validate_profile_duration(numeric_time)

        result = self.pipeline.run(time=time, accel=accel, height=height)
        if self.gait_res_key not in result:
            raise KeyError(
                f"SKDH pipeline output missing expected key '{self.gait_res_key}'."
            )
        return GaitResults(result[self.gait_res_key])

    def _build_pipeline(self) -> skdh.Pipeline:
        """Builds pipeline for multiday or treadmill profile."""
        pipeline = skdh.Pipeline()
        if not self.treadmill_profile:
            pipeline.add(
                self._build_step(
                    skdh.preprocessing.GetDayWindowIndices,
                    bases=[0],
                    periods=[24],
                )
            )
            pipeline.add(self._build_step(skdh.preprocessing.CalibrateAccelerometer))
            pipeline.add(self._build_step(skdh.context.PredictGaitLumbarLgbm))
            pipeline.add(
                self._build_step(
                    skdh.gait.GaitLumbar,
                    min_bout_time=8.0,
                    min_bout_duration=8.0,
                    min_steps=6,
                )
            )
        else:
            # Treadmill profile: avoid day-window and context-model gating.
            pipeline.add(
                self._build_step(
                    skdh.gait.GaitLumbar,
                    min_bout_time=4.0,
                    min_bout_duration=4.0,
                    min_steps=4,
                )
            )
        return pipeline

    def _build_step(self, step_constructor, **preferred_kwargs):
        accepted_kwargs: Dict[str, Any] = {}
        try:
            signature = inspect.signature(step_constructor)
            valid_keys = set(signature.parameters.keys())
            accepted_kwargs = {
                key: value
                for key, value in preferred_kwargs.items()
                if key in valid_keys
            }
        except Exception:
            accepted_kwargs = {}
        return step_constructor(**accepted_kwargs)

    def _coerce_time_for_validation(self, time) -> np.ndarray:
        time_array = np.asarray(time)
        if np.issubdtype(time_array.dtype, np.datetime64):
            return time_array.astype("datetime64[ns]").astype(np.int64) / 1e9
        try:
            return time_array.astype(float)
        except Exception:
            if len(time_array) > 0 and hasattr(time_array[0], "timestamp"):
                return np.array([timestamp.timestamp() for timestamp in time_array])
            raise ValueError("Unable to coerce time values for monotonic validation.")

    def _validate_time(self, numeric_time: np.ndarray, accel: np.ndarray) -> None:
        if len(numeric_time) == 0 or accel.shape[0] == 0:
            raise ValueError("IMU input is empty; unable to extract gait features.")
        if len(numeric_time) != accel.shape[0]:
            raise ValueError(
                "Time and accelerometer lengths do not match "
                f"({len(numeric_time)} vs {accel.shape[0]})."
            )
        time_delta = np.diff(numeric_time)
        if np.any(time_delta <= 0):
            raise ValueError("Time vector is not strictly increasing.")

    def _normalize_accel_units(self, accel: np.ndarray) -> np.ndarray:
        accel = accel.astype(float)
        if np.any(~np.isfinite(accel)):
            raise ValueError("Accelerometer array contains non-finite values.")
        norms = np.linalg.norm(accel, axis=1)
        median_norm = float(np.nanmedian(norms))
        if 0.25 <= median_norm <= 2.5:
            print("Detected accelerometer units in g; converting to m/s^2.")
            return accel * self.GRAVITY_M_PER_S2
        return accel

    def _validate_sensor_sanity(self, accel: np.ndarray) -> None:
        axis_std = np.nanstd(accel, axis=0)
        if np.all(axis_std < 1e-4):
            raise ValueError("Accelerometer signal variance is near zero on all axes.")

    def _normalize_height(self, height: float) -> float:
        normalized_height = float(height)
        if normalized_height <= 0:
            raise ValueError(f"Invalid user height value: {normalized_height}")
        if 100 <= normalized_height <= 250:
            print(
                f"Detected height in centimeters ({normalized_height}); "
                "converting to meters."
            )
            normalized_height = normalized_height / 100.0
        if normalized_height > 2.5:
            raise ValueError(
                f"User height {normalized_height}m is out of expected range after normalization."
            )
        return normalized_height

    def _validate_profile_duration(self, numeric_time: np.ndarray) -> None:
        duration_seconds = float(numeric_time[-1] - numeric_time[0])
        if self.treadmill_profile and duration_seconds > 3600:
            print(
                "Warning: treadmill profile enabled for recording longer than 1 hour. "
                "Consider multiday profile."
            )
        if not self.treadmill_profile and duration_seconds < 600:
            print(
                "Warning: multiday profile enabled for short recording (<10 minutes). "
                "Consider treadmill profile."
            )
