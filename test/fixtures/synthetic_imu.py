"""Synthetic minimal IMU records for unit tests that must not depend on real data."""

from __future__ import annotations

import numpy as np

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
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


def _triaxial_sensor(
    *,
    sensor_type: SensorType,
    unit: str,
    sampling_rate_hz: float = 100.0,
    duration_seconds: float = 12.0,
) -> SensorData:
    time = np.arange(0.0, duration_seconds, 1.0 / sampling_rate_hz)
    x = np.sin(2 * np.pi * 1.1 * time)
    y = np.sin(2 * np.pi * 0.9 * time + 0.1)
    z = np.sin(2 * np.pi * 1.0 * time + 0.2)
    return SensorData(
        data=[
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.ANTEROPOSTERIOR),
                SensorAxis(SensorCoordinateSystem.X),
                x,
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.MEDIOLATERAL),
                SensorAxis(SensorCoordinateSystem.Y),
                y,
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.VERTICAL),
                SensorAxis(SensorCoordinateSystem.Z),
                z,
            ),
        ],
        time=time,
        idle_mask=np.zeros_like(time),
        metadata=SensorMetadata(
            sensor_type=sensor_type,
            sampling_rate=sampling_rate_hz,
            unit=unit,
        ),
    )


def minimal_skdh_imu_data(imu_id: str = "imu_1") -> IMUData:
    """Accelerometer-only IMU suitable for SKDH backend requirement checks."""
    sensor = _triaxial_sensor(
        sensor_type=SensorType.ACCELEROMETER,
        unit="m/s^2",
    )
    metadata = IMUMetadata(
        imu_data_identifier=IMUDataIdentifier(imu_id),
        user_identifier=UserIdentifier("user_1"),
        instrument_identifier=InstrumentIdentifier("test_imu", "001"),
    )
    return IMUData(
        data=[sensor],
        metadata=metadata,
        start_time=float(sensor.time[0]),
        end_time=float(sensor.time[-1]),
    )


def minimal_mobgap_imu_data(imu_id: str = "imu_1") -> IMUData:
    """Accelerometer + gyroscope IMU suitable for MobGap requirement checks."""
    accel = _triaxial_sensor(
        sensor_type=SensorType.ACCELEROMETER,
        unit="m/s^2",
    )
    gyro = _triaxial_sensor(
        sensor_type=SensorType.GYROSCOPE,
        unit="rad/s",
    )
    metadata = IMUMetadata(
        imu_data_identifier=IMUDataIdentifier(imu_id),
        user_identifier=UserIdentifier("user_1"),
        instrument_identifier=InstrumentIdentifier("test_imu", "001"),
    )
    return IMUData(
        data=[accel, gyro],
        metadata=metadata,
        start_time=float(accel.time[0]),
        end_time=float(accel.time[-1]),
    )
