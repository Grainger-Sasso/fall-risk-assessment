from typing import Any, Dict

import numpy as np
import skdh

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.user.user_data import UserData
from src.data_processing.feature_processing.gait_feature_processor import (
    GaitFeatureProcessor,
)
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


class GaitFeatureExtractor:
    def __init__(self):
        self.pipeline: skdh.Pipeline = self._build_pipeline()
        self.feature_processor = GaitFeatureProcessor()
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
        height = user_data.clinical_demographic_data.height.value
        return self.pipeline.run(time=time, accel=accel, height=height)[self.gait_res_key]

    def _build_pipeline(self) -> skdh.Pipeline:
        """Builds pipeline to extract gait features"""
        pipeline = skdh.Pipeline()
        pipeline.add(skdh.preprocessing.GetDayWindowIndices(bases=[0], periods=[24]))
        pipeline.add(skdh.preprocessing.CalibrateAccelerometer())
        pipeline.add(skdh.context.PredictGaitLumbarLgbm())
        pipeline.add(skdh.gait.GaitLumbar())
        return pipeline
