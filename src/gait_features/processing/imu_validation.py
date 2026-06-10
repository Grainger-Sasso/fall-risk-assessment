import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.data_model.instrument_specifications.sensor_specifications import SensorSpecification
from src.data_types.instrument.sensor_type import SensorType
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.processing.signal_preprocessing import (
    AccelOutputUnit,
    BackendSignalProfile,
    GyroOutputUnit,
    canonicalize_accel_unit,
    canonicalize_gyro_unit,
    convert_accel,
    convert_gyro,
    infer_accel_unit,
)
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


class ImuValidationMixin:
    GRAVITY_M_PER_S2 = 9.80665

    @staticmethod
    def select_triaxial_accelerometer_sensor(imu_data: IMUData) -> SensorData:
        if not imu_data.data:
            raise ValueError("IMUData contains no sensor streams.")
        for sensor_data in imu_data.data:
            try:
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X)
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y)
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z)
                return sensor_data
            except Exception:
                continue
        raise ValueError("No triaxial accelerometer stream found in IMUData.")

    @staticmethod
    def select_triaxial_gyroscope_sensor(imu_data: IMUData) -> Optional[SensorData]:
        for sensor_data in imu_data.data:
            if sensor_data.metadata.sensor_type != SensorType.GYROSCOPE:
                continue
            try:
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X)
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y)
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z)
                return sensor_data
            except Exception:
                continue
        return None

    @staticmethod
    def build_accel_matrix(sensor_data: SensorData) -> np.ndarray:
        return np.column_stack(
            (
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data,
            )
        )

    @staticmethod
    def build_gyro_matrix(sensor_data: SensorData) -> np.ndarray:
        return np.column_stack(
            (
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data,
            )
        )

    def prepare_accelerometer(
        self,
        imu_data: IMUData,
        instrument_specifications: Optional[IMUSpecifications],
        treadmill_profile: bool,
        accel_output_unit: AccelOutputUnit,
    ) -> Tuple[SensorData, np.ndarray, Any]:
        sensor_data = self.select_triaxial_accelerometer_sensor(imu_data)
        time = sensor_data.time
        accel = self.build_accel_matrix(sensor_data)
        numeric_time = self._coerce_time_for_validation(time)
        self._validate_time(numeric_time=numeric_time, accel=accel)
        inferred_sampling_rate = self._estimate_sampling_rate_hz(numeric_time)
        inferred_unit = infer_accel_unit(accel)
        self._validate_sensor_metadata(
            sensor_data=sensor_data,
            inferred_sampling_rate=inferred_sampling_rate,
            inferred_unit=inferred_unit,
        )
        if instrument_specifications is not None:
            self._validate_instrument_specification(
                instrument_specifications=instrument_specifications,
                sensor_data=sensor_data,
                raw_accel=accel,
                inferred_sampling_rate=inferred_sampling_rate,
                inferred_unit=inferred_unit,
            )
        accel = convert_accel(
            accel=accel,
            source_unit=inferred_unit,
            target_unit=accel_output_unit,
        )
        self._validate_sensor_sanity(accel=accel)
        self._validate_profile_duration(numeric_time, treadmill_profile=treadmill_profile)
        return sensor_data, accel, time

    def prepare_accelerometer_for_backend(
        self,
        imu_data: IMUData,
        instrument_specifications: Optional[IMUSpecifications],
        treadmill_profile: bool,
        backend: GaitExtractionBackendId,
    ) -> Tuple[SensorData, np.ndarray, Any]:
        profile = BackendSignalProfile.for_backend(backend)
        return self.prepare_accelerometer(
            imu_data=imu_data,
            instrument_specifications=instrument_specifications,
            treadmill_profile=treadmill_profile,
            accel_output_unit=profile.accel_unit,
        )

    def prepare_gyroscope(
        self,
        imu_data: IMUData,
        gyro_output_unit: GyroOutputUnit,
    ) -> Tuple[SensorData, np.ndarray]:
        sensor_data = self.select_triaxial_gyroscope_sensor(imu_data)
        if sensor_data is None:
            raise ValueError("No triaxial gyroscope stream found in IMUData.")
        gyro = self.build_gyro_matrix(sensor_data)
        declared_unit = canonicalize_gyro_unit(sensor_data.metadata.unit)
        if declared_unit is None:
            self._warn(
                f"Unknown gyroscope unit '{sensor_data.metadata.unit}'; "
                f"assuming deg/s for conversion to {gyro_output_unit.value}."
            )
            declared_unit = GyroOutputUnit.DEG_S.value
        gyro = convert_gyro(
            gyro=gyro,
            source_unit=declared_unit,
            target_unit=gyro_output_unit,
        )
        return sensor_data, gyro

    def prepare_gyroscope_for_backend(
        self,
        imu_data: IMUData,
        backend: GaitExtractionBackendId,
    ) -> Tuple[SensorData, np.ndarray]:
        profile = BackendSignalProfile.for_backend(backend)
        return self.prepare_gyroscope(
            imu_data=imu_data,
            gyro_output_unit=profile.gyro_unit,
        )

    def prepare_accel_matrix_mps2(self, sensor_data: SensorData) -> np.ndarray:
        accel = self.build_accel_matrix(sensor_data)
        inferred_unit = infer_accel_unit(accel)
        return convert_accel(
            accel=accel,
            source_unit=inferred_unit,
            target_unit=AccelOutputUnit.MPS2,
        )

    @staticmethod
    def normalize_height(height: float) -> float:
        normalized_height = float(height)
        if normalized_height <= 0:
            raise ValueError(f"Invalid user height value: {normalized_height}")
        if 100 <= normalized_height <= 250:
            print(
                f"Detected height in centimeters ({normalized_height}); converting to meters."
            )
            normalized_height = normalized_height / 100.0
        if normalized_height > 2.5:
            raise ValueError(
                f"User height {normalized_height}m is out of expected range after normalization."
            )
        return normalized_height

    @staticmethod
    def default_sensor_height_m(height_m: float) -> float:
        return 0.55 * height_m

    @staticmethod
    def build_step(step_constructor, **preferred_kwargs):
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
        if np.any(np.diff(numeric_time) <= 0):
            raise ValueError("Time vector is not strictly increasing.")

    def _validate_sensor_sanity(self, accel: np.ndarray) -> None:
        if np.all(np.nanstd(accel, axis=0) < 1e-4):
            raise ValueError("Accelerometer signal variance is near zero on all axes.")

    def _validate_profile_duration(self, numeric_time: np.ndarray, treadmill_profile: bool) -> None:
        duration_seconds = float(numeric_time[-1] - numeric_time[0])
        if treadmill_profile and duration_seconds > 3600:
            print(
                "Warning: treadmill profile enabled for recording longer than 1 hour. "
                "Consider multiday profile."
            )
        if not treadmill_profile and duration_seconds < 600:
            print(
                "Warning: multiday profile enabled for short recording (<10 minutes). "
                "Consider treadmill profile."
            )

    def _validate_sensor_metadata(
        self,
        sensor_data: SensorData,
        inferred_sampling_rate: float,
        inferred_unit: str,
    ) -> None:
        declared_rate = float(sensor_data.metadata.sampling_rate)
        if declared_rate <= 0:
            raise ValueError(
                f"Sensor metadata sampling_rate must be positive, found {declared_rate}."
            )
        max_allowed_delta = max(1.0, declared_rate * 0.05)
        if abs(declared_rate - inferred_sampling_rate) > max_allowed_delta:
            raise ValueError(
                "Sensor metadata sampling_rate mismatch: "
                f"declared={declared_rate:.4f}Hz inferred={inferred_sampling_rate:.4f}Hz."
            )

        declared_unit = self._canonicalize_unit(sensor_data.metadata.unit)
        if declared_unit is None:
            raise ValueError(
                f"Unsupported sensor metadata unit '{sensor_data.metadata.unit}'."
            )
        if declared_unit != inferred_unit:
            raise ValueError(
                "Sensor metadata unit mismatch: "
                f"declared={declared_unit} inferred={inferred_unit}."
            )

    def _validate_instrument_specification(
        self,
        instrument_specifications: IMUSpecifications,
        sensor_data: SensorData,
        raw_accel: np.ndarray,
        inferred_sampling_rate: float,
        inferred_unit: str,
    ) -> None:
        self._validate_instrument_specification_shape(instrument_specifications)
        sensor_type = sensor_data.metadata.sensor_type
        if sensor_type != SensorType.ACCELEROMETER:
            return

        try:
            sensor_spec = instrument_specifications.get_specification_by_sensor_type(
                sensor_type
            )
        except Exception as exc:
            raise ValueError(
                f"Instrument specification missing sensor type {sensor_type.value}."
            ) from exc

        spec_unit = None
        if sensor_spec.units is None:
            self._warn(
                "Instrument specification field 'units' is None; skipping unit validation "
                f"for sensor '{sensor_spec.sensor_name}'."
            )
        else:
            spec_unit = self._canonicalize_unit(sensor_spec.units)
            if spec_unit is None:
                raise ValueError(
                    f"Unsupported instrument specification unit '{sensor_spec.units}'."
                )
            if spec_unit != inferred_unit:
                raise ValueError(
                    "Instrument specification unit mismatch: "
                    f"spec={spec_unit} inferred={inferred_unit}."
                )

            declared_meta_unit = self._canonicalize_unit(sensor_data.metadata.unit)
            if declared_meta_unit != spec_unit:
                raise ValueError(
                    "Instrument specification and sensor metadata unit mismatch: "
                    f"spec={spec_unit} metadata={declared_meta_unit}."
                )

        if sensor_spec.sampling_rate is None:
            self._warn(
                "Instrument specification field 'sampling_rate' is None; skipping sampling-rate "
                f"validation for sensor '{sensor_spec.sensor_name}'."
            )
        else:
            sampling_rate_value = float(sensor_spec.sampling_rate)
            max_allowed_delta = max(1.0, sampling_rate_value * 0.05)
            if abs(sampling_rate_value - inferred_sampling_rate) > max_allowed_delta:
                raise ValueError(
                    "Instrument specification sampling_rate mismatch: "
                    f"spec={sampling_rate_value:.4f}Hz inferred={inferred_sampling_rate:.4f}Hz."
                )

        if sensor_spec.range is None:
            self._warn(
                "Instrument specification field 'range' is None; skipping range validation "
                f"for sensor '{sensor_spec.sensor_name}'."
            )
        else:
            range_min, range_max = sensor_spec.range
            if range_min is None or range_max is None:
                self._warn(
                    "Instrument specification field 'range' has None bounds; skipping range "
                    f"validation for sensor '{sensor_spec.sensor_name}'."
                )
            elif (
                not np.isfinite(float(range_min))
                or not np.isfinite(float(range_max))
                or float(range_min) >= float(range_max)
            ):
                raise ValueError(
                    "Instrument specification range must be finite and increasing."
                )
            else:
                observed_min = float(np.nanmin(raw_accel))
                observed_max = float(np.nanmax(raw_accel))
                range_min = float(range_min)
                range_max = float(range_max)
                tolerance = max(0.1, abs(range_max - range_min) * 0.05)
                if observed_min < range_min - tolerance or observed_max > range_max + tolerance:
                    range_unit = spec_unit or inferred_unit
                    raise ValueError(
                        "Instrument specification range mismatch: "
                        f"observed=[{observed_min:.4f}, {observed_max:.4f}] "
                        f"spec=[{range_min:.4f}, {range_max:.4f}] in {range_unit}."
                    )

    def _validate_instrument_specification_shape(
        self, instrument_specifications: IMUSpecifications
    ) -> None:
        for sensor_spec in instrument_specifications.sensor_specifications:
            self._validate_single_sensor_spec(sensor_spec)

    def _validate_single_sensor_spec(self, sensor_spec: SensorSpecification) -> None:
        sensor_name = sensor_spec.sensor_name
        if sensor_name is None or not isinstance(sensor_name, str) or not sensor_name.strip():
            self._warn(
                "Instrument specification field 'sensor_name' is missing; using "
                "'<unknown_sensor>' in validation messages."
            )
            sensor_name = "<unknown_sensor>"

        self._validate_positive_finite_numeric(
            sensor_spec.sampling_rate, "sampling_rate", sensor_name
        )
        self._validate_positive_finite_numeric(
            sensor_spec.sensitivity, "sensitivity", sensor_name
        )
        if sensor_spec.resolution is None:
            self._warn(
                f"Instrument specification field 'resolution' is None for sensor '{sensor_name}'."
            )
        elif not isinstance(sensor_spec.resolution, int) or sensor_spec.resolution <= 0:
            raise ValueError(
                f"Instrument specification field 'resolution' is not a positive integer "
                f"for sensor '{sensor_name}'."
            )
        self._validate_nonnegative_finite_numeric(
            sensor_spec.noise_density, "noise_density", sensor_name
        )
        self._validate_nonnegative_finite_numeric(
            sensor_spec.bias_stability, "bias_stability", sensor_name
        )
        self._validate_nonnegative_finite_numeric(
            sensor_spec.alignment_error, "alignment_error", sensor_name
        )
        self._validate_nonnegative_finite_numeric(
            sensor_spec.cross_axis_sensitivity,
            "cross_axis_sensitivity",
            sensor_name,
        )
        self._validate_nonnegative_finite_numeric(
            sensor_spec.power_consumption, "power_consumption", sensor_name
        )
        self._validate_nonnegative_finite_numeric(
            sensor_spec.mass, "mass", sensor_name
        )
        if sensor_spec.physical_size is None:
            self._warn(
                f"Instrument specification field 'physical_size' is None for sensor '{sensor_name}'."
            )
            return

        if (
            not isinstance(sensor_spec.physical_size, tuple)
            or len(sensor_spec.physical_size) != 3
        ):
            raise ValueError(
                "Instrument specification field 'physical_size' must be a 3-tuple "
                f"for sensor '{sensor_name}'."
            )

        for axis_name, value in zip(("length", "width", "height"), sensor_spec.physical_size):
            if value is None:
                self._warn(
                    "Instrument specification field "
                    f"'physical_size.{axis_name}' is None for sensor '{sensor_name}'."
                )
                continue
            numeric_value = self._coerce_float_field(
                value=value,
                field_name=f"physical_size.{axis_name}",
                sensor_name=sensor_name,
            )
            if numeric_value <= 0:
                raise ValueError(
                    "Instrument specification field "
                    f"'physical_size.{axis_name}' must be positive for sensor '{sensor_name}'."
                )

    def _coerce_float_field(self, value, field_name: str, sensor_name: str) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Instrument specification field '{field_name}' must be numeric "
                f"for sensor '{sensor_name}'."
            ) from exc
        if not np.isfinite(numeric_value):
            raise ValueError(
                f"Instrument specification field '{field_name}' must be finite "
                f"for sensor '{sensor_name}'."
            )
        return numeric_value

    def _canonicalize_unit(self, unit_value: str) -> Optional[str]:
        return canonicalize_accel_unit(unit_value)

    def _validate_positive_finite_numeric(
        self, value, field_name: str, sensor_name: str
    ) -> None:
        if value is None:
            self._warn(
                f"Instrument specification field '{field_name}' is None for sensor '{sensor_name}'."
            )
            return
        numeric_value = self._coerce_float_field(value, field_name, sensor_name)
        if numeric_value <= 0:
            raise ValueError(
                f"Instrument specification field '{field_name}' must be positive and finite "
                f"for sensor '{sensor_name}'."
            )

    def _validate_nonnegative_finite_numeric(
        self, value, field_name: str, sensor_name: str
    ) -> None:
        if value is None:
            self._warn(
                f"Instrument specification field '{field_name}' is None for sensor '{sensor_name}'."
            )
            return
        numeric_value = self._coerce_float_field(value, field_name, sensor_name)
        if numeric_value < 0:
            raise ValueError(
                f"Instrument specification field '{field_name}' must be non-negative and finite "
                f"for sensor '{sensor_name}'."
            )

    def _warn(self, message: str) -> None:
        print(f"[WARNING] {message}")

    @staticmethod
    def _estimate_sampling_rate_hz(numeric_time: np.ndarray) -> float:
        if numeric_time.size < 2:
            return 1.0
        dt = np.diff(numeric_time)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return 1.0
        return float(1.0 / np.median(dt))
