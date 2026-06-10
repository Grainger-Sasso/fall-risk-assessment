from src.data_model.data.imu.imu_data import IMUData
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.processing.imu_validation import ImuValidationMixin


class BackendRequirementsError(ValueError):
    """Raised when IMU data does not meet a gait backend's sensor requirements."""


class BackendRequirementsValidator(ImuValidationMixin):
    def validate(self, imu_data: IMUData, backend: GaitExtractionBackendId) -> None:
        if backend == GaitExtractionBackendId.MOBGAP:
            self._validate_mobgap_requirements(imu_data)
        elif backend == GaitExtractionBackendId.SKDH:
            self._validate_skdh_requirements(imu_data)
        else:
            raise ValueError(f"Unsupported extraction backend: {backend.value}")

    def _validate_skdh_requirements(self, imu_data: IMUData) -> None:
        self.select_triaxial_accelerometer_sensor(imu_data)

    def _validate_mobgap_requirements(self, imu_data: IMUData) -> None:
        self.select_triaxial_accelerometer_sensor(imu_data)
        gyro_sensor = self.select_triaxial_gyroscope_sensor(imu_data)
        if gyro_sensor is None:
            raise BackendRequirementsError(
                "MobGap extraction requires a triaxial gyroscope stream in IMUData. "
                "This recording has accelerometer data only; use the SKDH backend "
                "or provide IMU data that includes gyroscope measurements."
            )
        accel_sensor = self.select_triaxial_accelerometer_sensor(imu_data)
        if len(accel_sensor.time) != len(gyro_sensor.time):
            raise BackendRequirementsError(
                "MobGap extraction requires accelerometer and gyroscope streams "
                f"with equal sample counts ({len(accel_sensor.time)} vs "
                f"{len(gyro_sensor.time)})."
            )


def validate_extraction_requirements(
    imu_data: IMUData,
    backend: GaitExtractionBackendId,
) -> None:
    BackendRequirementsValidator().validate(imu_data=imu_data, backend=backend)
