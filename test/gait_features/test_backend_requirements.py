import unittest

import numpy as np

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
from src.gait_features.backends.mobgap.backend import MobgapBackend
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.processing.backend_requirements import (
    BackendRequirementsError,
    validate_extraction_requirements,
)
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import AnatomicalAxis
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)
from test.gait_features.test_gait_feature_extractor_metadata_validation import (
    _make_imu_data,
    _make_user_data,
)


def _make_imu_with_gyro() -> IMUData:
    accel = _make_imu_data(unit="g", sampling_rate=100.0)
    time = np.asarray(accel.data[0].time, dtype=float)
    gyro_stream = SensorData(
        data=[
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.ANTEROPOSTERIOR),
                SensorAxis(SensorCoordinateSystem.X),
                np.zeros_like(time),
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.MEDIOLATERAL),
                SensorAxis(SensorCoordinateSystem.Y),
                np.zeros_like(time),
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.VERTICAL),
                SensorAxis(SensorCoordinateSystem.Z),
                np.zeros_like(time),
            ),
        ],
        time=time,
        idle_mask=np.zeros_like(time),
        metadata=SensorMetadata(
            sensor_type=SensorType.GYROSCOPE,
            sampling_rate=100.0,
            unit="deg/s",
        ),
    )
    return IMUData(
        data=[accel.data[0], gyro_stream],
        metadata=accel.metadata,
        start_time=float(time[0]),
        end_time=float(time[-1]),
    )


class TestBackendRequirements(unittest.TestCase):
    def test_skdh_requires_triaxial_accelerometer(self):
        validate_extraction_requirements(
            imu_data=_make_imu_data(unit="g", sampling_rate=100.0),
            backend=GaitExtractionBackendId.SKDH,
        )

    def test_mobgap_rejects_accel_only_imu(self):
        with self.assertRaises(BackendRequirementsError):
            validate_extraction_requirements(
                imu_data=_make_imu_data(unit="g", sampling_rate=100.0),
                backend=GaitExtractionBackendId.MOBGAP,
            )

    def test_mobgap_accepts_accel_and_gyro(self):
        validate_extraction_requirements(
            imu_data=_make_imu_with_gyro(),
            backend=GaitExtractionBackendId.MOBGAP,
        )

    def test_mobgap_backend_raises_before_pipeline(self):
        backend = MobgapBackend()
        with self.assertRaises(BackendRequirementsError):
            backend.extract(
                imu_data=_make_imu_data(unit="g", sampling_rate=100.0),
                user_data=_make_user_data(),
            )


if __name__ == "__main__":
    unittest.main()
