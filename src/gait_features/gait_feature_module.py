import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import skdh
except ModuleNotFoundError:
    skdh = None

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.data_model.instrument_specifications.sensor_specifications import SensorSpecification
from src.data_types.feature.feature_type import FeatureType
from src.data_types.feature.feature_units import get_feature_unit
from src.data_types.instrument.sensor_type import SensorType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.identifier import IdentifierGenerator
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


class GaitResults:
    def __init__(self, results: Dict[str, Any]):
        self.data: Dict[str, Any] = results


class GaitFeatureExtractor:
    GRAVITY_M_PER_S2 = 9.80665
    UNIT_ALIASES = {
        "g": "g",
        "gravities": "g",
        "gravity": "g",
        "m/s^2": "m/s2",
        "m/s2": "m/s2",
        "ms^-2": "m/s2",
        "m/s/s": "m/s2",
    }

    def __init__(self, treadmill_profile: bool = False):
        self.treadmill_profile = treadmill_profile
        self.pipeline = self._build_pipeline()
        self.gait_res_key = "GaitLumbar"

    def extract_gait_features(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications] = None,
    ) -> GaitResults:
        if self.pipeline is None:
            raise ModuleNotFoundError(
                "scikit-digital-health (skdh) is required for gait feature extraction."
            )
        sensor_data = self._select_triaxial_accelerometer_sensor(imu_data)
        time = sensor_data.time
        accel = np.column_stack(
            (
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data,
            )
        )
        numeric_time = self._coerce_time_for_validation(time)
        self._validate_time(numeric_time=numeric_time, accel=accel)
        inferred_sampling_rate = self._estimate_sampling_rate_hz(numeric_time)
        inferred_unit = self._infer_accel_unit(accel)
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
        accel = self._normalize_accel_units(accel, inferred_unit=inferred_unit)
        self._validate_sensor_sanity(accel=accel)
        height = self._normalize_height(user_data.clinical_demographic_data.height.value)
        self._validate_profile_duration(numeric_time)

        result = self.pipeline.run(time=time, accel=accel, height=height)
        if self.gait_res_key not in result:
            raise KeyError(
                f"SKDH pipeline output missing expected key '{self.gait_res_key}'."
            )
        return GaitResults(result[self.gait_res_key])

    def _build_pipeline(self):
        if skdh is None:
            return None
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
        if np.any(np.diff(numeric_time) <= 0):
            raise ValueError("Time vector is not strictly increasing.")

    def _normalize_accel_units(self, accel: np.ndarray, inferred_unit: str) -> np.ndarray:
        accel = accel.astype(float)
        if np.any(~np.isfinite(accel)):
            raise ValueError("Accelerometer array contains non-finite values.")
        if inferred_unit == "g":
            print("Detected accelerometer units in g; converting to m/s^2.")
            return accel * self.GRAVITY_M_PER_S2
        return accel

    def _validate_sensor_sanity(self, accel: np.ndarray) -> None:
        if np.all(np.nanstd(accel, axis=0) < 1e-4):
            raise ValueError("Accelerometer signal variance is near zero on all axes.")

    def _normalize_height(self, height: float) -> float:
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

    def _infer_accel_unit(self, accel: np.ndarray) -> str:
        median_norm = float(np.nanmedian(np.linalg.norm(accel, axis=1)))
        if 0.25 <= median_norm <= 2.5:
            return "g"
        return "m/s2"

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
        normalized = str(unit_value).strip().lower().replace(" ", "")
        return self.UNIT_ALIASES.get(normalized)

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

    @staticmethod
    def _select_triaxial_accelerometer_sensor(imu_data: IMUData) -> SensorData:
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


class StrideFeatureMapper:
    """Maps SKDH event-level outputs to stride-basis BoutFeatures tensors."""

    def __init__(self, feature_types: Optional[List[FeatureType]] = None):
        self.feature_types = feature_types or FeatureType.get_stride_feature_types()

    def build(
        self,
        gait_results: GaitResults,
        bout_event_ranges: List[Tuple[int, int]],
        bout_time_ranges: List[Tuple[float, float]],
    ) -> BoutFeatures:
        num_bouts = len(bout_event_ranges)
        num_features = len(self.feature_types)
        max_samples = self._resolve_max_samples(bout_event_ranges)
        features = np.full((num_bouts, num_features, max_samples), np.nan, dtype=float)

        for bout_index, (event_start, event_end) in enumerate(bout_event_ranges):
            for feature_index, feature_type in enumerate(self.feature_types):
                values = self._extract_feature_slice(
                    gait_results=gait_results,
                    feature_type=feature_type,
                    start_index=event_start,
                    end_index=event_end,
                    fallback_length=max_samples,
                )
                if values.size > 0:
                    features[bout_index, feature_index, : values.size] = values

        if max_samples == 0:
            sample_starts = np.array([], dtype=float)
            sample_ends = np.array([], dtype=float)
        else:
            sample_starts = np.arange(max_samples, dtype=float)
            sample_ends = sample_starts + 1.0

        return BoutFeatures(
            sample_basis=SampleBasis.STRIDE,
            features=features,
            bout_starts=np.array([window[0] for window in bout_time_ranges], dtype=float),
            bout_ends=np.array([window[1] for window in bout_time_ranges], dtype=float),
            feature_names=self.feature_types,
            sample_starts=sample_starts,
            sample_ends=sample_ends,
            units=[get_feature_unit(feature_type) for feature_type in self.feature_types],
        )

    @staticmethod
    def _resolve_max_samples(bout_event_ranges: List[Tuple[int, int]]) -> int:
        if not bout_event_ranges:
            return 0
        return max(1, max(0, max(end - start for start, end in bout_event_ranges)))

    def _extract_feature_slice(
        self,
        gait_results: GaitResults,
        feature_type: FeatureType,
        start_index: int,
        end_index: int,
        fallback_length: int,
    ) -> np.ndarray:
        raw_values = gait_results.data.get(feature_type.value)
        if raw_values is None:
            return np.array([np.nan], dtype=float) if start_index == end_index and fallback_length > 0 else np.array([], dtype=float)
        values = np.asarray(raw_values, dtype=float)
        if values.size == 0:
            return np.array([np.nan], dtype=float) if start_index == end_index and fallback_length > 0 else np.array([], dtype=float)
        bounded_start = max(0, min(start_index, values.size))
        bounded_end = max(bounded_start, min(end_index, values.size))
        sliced = values[bounded_start:bounded_end]
        if sliced.size == 0 and fallback_length > 0:
            return np.array([np.nan], dtype=float)
        return sliced


class EpochFeatureGenerator:
    """Generates sliding-window epoch features from triaxial acceleration."""

    def __init__(
        self,
        window_seconds: float = 8.0,
        overlap_seconds: float = 2.0,
        feature_types: Optional[List[FeatureType]] = None,
    ):
        if window_seconds < 5.0 or window_seconds > 10.0:
            raise ValueError("window_seconds must be between 5 and 10 seconds.")
        if overlap_seconds <= 0 or overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be > 0 and strictly less than window_seconds.")
        self.window_seconds = float(window_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.feature_types = feature_types or FeatureType.get_epoch_feature_types()

    def build(
        self,
        sensor_data: SensorData,
        bout_sample_ranges: List[Tuple[int, int]],
        bout_time_ranges: List[Tuple[float, float]],
    ) -> BoutFeatures:
        x = np.asarray(sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data, dtype=float)
        y = np.asarray(sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data, dtype=float)
        z = np.asarray(sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data, dtype=float)
        sampling_rate_hz = self._estimate_sampling_rate_hz(np.asarray(sensor_data.time, dtype=float))

        bout_matrices: List[np.ndarray] = []
        max_windows = 0
        for sample_start, sample_end in bout_sample_ranges:
            matrix = self._build_single_bout_feature_matrix(
                x=x,
                y=y,
                z=z,
                start_index=sample_start,
                end_index=sample_end,
                sampling_rate_hz=sampling_rate_hz,
            )
            bout_matrices.append(matrix)
            max_windows = max(max_windows, matrix.shape[1])

        features = np.full((len(bout_matrices), len(self.feature_types), max_windows), np.nan, dtype=float)
        for bout_index, matrix in enumerate(bout_matrices):
            if matrix.size > 0:
                features[bout_index, :, : matrix.shape[1]] = matrix

        if max_windows == 0:
            sample_starts = np.array([], dtype=float)
            sample_ends = np.array([], dtype=float)
        else:
            step_seconds = self.window_seconds - self.overlap_seconds
            sample_starts = np.arange(max_windows, dtype=float) * step_seconds
            sample_ends = sample_starts + self.window_seconds

        return BoutFeatures(
            sample_basis=SampleBasis.EPOCH,
            features=features,
            bout_starts=np.array([window[0] for window in bout_time_ranges], dtype=float),
            bout_ends=np.array([window[1] for window in bout_time_ranges], dtype=float),
            feature_names=self.feature_types,
            sample_starts=sample_starts,
            sample_ends=sample_ends,
            units=[get_feature_unit(feature_type) for feature_type in self.feature_types],
        )

    def _build_single_bout_feature_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        start_index: int,
        end_index: int,
        sampling_rate_hz: float,
    ) -> np.ndarray:
        bounded_start = max(0, start_index)
        bounded_end = max(bounded_start, min(end_index, x.size))
        if bounded_end - bounded_start <= 1:
            return np.full((len(self.feature_types), 1), np.nan, dtype=float)

        windowed_values = []
        for window_start, window_end in self._resolve_windows(bounded_start, bounded_end, sampling_rate_hz):
            x_window = x[window_start:window_end]
            y_window = y[window_start:window_end]
            z_window = z[window_start:window_end]
            feature_values = self._compute_window_features(x_window, y_window, z_window, sampling_rate_hz)
            vector = [feature_values.get(feature_type, np.nan) for feature_type in self.feature_types]
            windowed_values.append(vector)

        if not windowed_values:
            return np.full((len(self.feature_types), 1), np.nan, dtype=float)
        return np.asarray(windowed_values, dtype=float).T

    def _resolve_windows(self, start_index: int, end_index: int, sampling_rate_hz: float) -> List[Tuple[int, int]]:
        window_samples = max(2, int(round(self.window_seconds * sampling_rate_hz)))
        step_samples = max(1, int(round((self.window_seconds - self.overlap_seconds) * sampling_rate_hz)))
        if end_index - start_index <= window_samples:
            return [(start_index, end_index)]
        windows: List[Tuple[int, int]] = []
        cursor = start_index
        while cursor + window_samples <= end_index:
            windows.append((cursor, cursor + window_samples))
            cursor += step_samples
        return windows or [(start_index, end_index)]

    def _compute_window_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        sampling_rate_hz: float,
    ) -> Dict[FeatureType, float]:
        magnitude = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        signal_map = {
            "vertical": z,
            "mediolateral": y,
            "anteroposterior": x,
            "magnitude": magnitude,
        }
        features: Dict[FeatureType, float] = {}
        for axis_key, signal in signal_map.items():
            self._populate_time_domain(features, axis_key, signal, sampling_rate_hz)
            self._populate_frequency_domain(features, axis_key, signal, sampling_rate_hz)
        return features

    def _populate_time_domain(self, target: Dict[FeatureType, float], axis_key: str, signal: np.ndarray, sampling_rate_hz: float) -> None:
        demeaned = signal - np.nanmean(signal)
        jerk = np.diff(signal) * sampling_rate_hz if signal.size > 1 else np.array([np.nan])
        metric_values = {
            "mean": float(np.nanmean(signal)),
            "std": float(np.nanstd(signal)),
            "rms": float(np.sqrt(np.nanmean(np.square(signal)))),
            "sma": float(np.nanmean(np.abs(signal))),
            "zero_crossing_rate": self._zero_crossing_rate(demeaned),
            "jerk_rms": float(np.sqrt(np.nanmean(np.square(jerk)))),
        }
        for metric_name, metric_value in metric_values.items():
            target[FeatureType.get_epoch_feature_type(axis_key, metric_name)] = metric_value

    def _populate_frequency_domain(self, target: Dict[FeatureType, float], axis_key: str, signal: np.ndarray, sampling_rate_hz: float) -> None:
        freqs, power = self._power_spectrum(signal, sampling_rate_hz)
        if freqs.size == 0 or power.size == 0:
            derived = {
                "dominant_frequency": np.nan,
                "spectral_entropy": np.nan,
                "bandpower_low": np.nan,
                "bandpower_high": np.nan,
            }
        else:
            power_sum = float(np.sum(power))
            normalized = power / power_sum if power_sum > 0 else np.zeros_like(power)
            derived = {
                "dominant_frequency": float(freqs[np.argmax(power)]),
                "spectral_entropy": float(-np.sum(normalized * np.log2(normalized + 1e-12))) if power_sum > 0 else np.nan,
                "bandpower_low": self._bandpower(freqs, power, 0.3, 1.0),
                "bandpower_high": self._bandpower(freqs, power, 1.0, 8.0),
            }
        for metric_name, metric_value in derived.items():
            target[FeatureType.get_epoch_feature_type(axis_key, metric_name)] = float(metric_value)

    @staticmethod
    def _estimate_sampling_rate_hz(time_values: np.ndarray) -> float:
        if time_values.size < 2:
            return 1.0
        dt = np.diff(time_values)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        return float(1.0 / np.median(dt)) if dt.size > 0 else 1.0

    @staticmethod
    def _zero_crossing_rate(signal: np.ndarray) -> float:
        if signal.size <= 1:
            return np.nan
        return float(np.sum(np.diff(np.sign(signal)) != 0) / max(1, signal.size - 1))

    @staticmethod
    def _power_spectrum(signal: np.ndarray, sampling_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
        if signal.size < 4 or sampling_rate_hz <= 0:
            return np.array([]), np.array([])
        centered = signal - np.nanmean(signal)
        fft = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate_hz)
        power = np.square(np.abs(fft))
        mask = (freqs >= 0.3) & (freqs <= 10.0)
        return freqs[mask], power[mask]

    @staticmethod
    def _bandpower(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return np.nan
        return float(np.trapezoid(power[mask], freqs[mask]))


@dataclass
class BoutRange:
    sample_start: int
    sample_end: int
    event_start: int
    event_end: int
    start_time: float
    end_time: float


class RecordFeatureGenerationBuilder:
    """Builds RecordFeatures from SKDH gait results and IMU signal windows."""

    def __init__(
        self,
        window_seconds: float = 8.0,
        overlap_seconds: float = 2.0,
        feature_id_generator: Optional[IdentifierGenerator[FeatureIdentifier]] = None,
    ):
        self.feature_id_generator = feature_id_generator or IdentifierGenerator("feature", FeatureIdentifier)
        self.stride_mapper = StrideFeatureMapper()
        self.epoch_generator = EpochFeatureGenerator(window_seconds=window_seconds, overlap_seconds=overlap_seconds)

    def build(
        self,
        gait_results: GaitResults,
        imu_data: IMUData,
        user_data: UserData,
        treadmill_profile: bool = False,
    ) -> RecordFeatures:
        sensor_data = self._select_accelerometer_sensor(imu_data)
        bout_ranges = self._resolve_bout_ranges(gait_results=gait_results, sensor_data=sensor_data, treadmill_profile=treadmill_profile)
        stride_features = self.stride_mapper.build(
            gait_results=gait_results,
            bout_event_ranges=[(item.event_start, item.event_end) for item in bout_ranges],
            bout_time_ranges=[(item.start_time, item.end_time) for item in bout_ranges],
        )
        epoch_features = self.epoch_generator.build(
            sensor_data=sensor_data,
            bout_sample_ranges=[(item.sample_start, item.sample_end) for item in bout_ranges],
            bout_time_ranges=[(item.start_time, item.end_time) for item in bout_ranges],
        )
        metadata = FeatureMetadata(
            feature_identifier=self.feature_id_generator.generate_identifier(),
            user_identifier=user_data.user_identifier,
            imu_data_identifier=imu_data.metadata.imu_data_identifier,
        )
        return RecordFeatures(epoch_features=epoch_features, stride_features=stride_features, feature_metadata=metadata)

    def _resolve_bout_ranges(self, gait_results: GaitResults, sensor_data: SensorData, treadmill_profile: bool) -> List[BoutRange]:
        day_values = gait_results.data.get("Day N")
        bout_values = gait_results.data.get("Bout N")
        ic_time_values = gait_results.data.get("IC Time")
        numeric_time = np.asarray(sensor_data.time, dtype=float)
        if day_values is None or bout_values is None or ic_time_values is None:
            return [self._full_recording_bout(sensor_data)]
        day_array = np.asarray(day_values)
        bout_array = np.asarray(bout_values)
        ic_seconds = self._to_seconds(np.asarray(ic_time_values))
        if day_array.size == 0 or bout_array.size == 0 or ic_seconds.size == 0:
            return [self._full_recording_bout(sensor_data)]

        length = min(day_array.size, bout_array.size, ic_seconds.size)
        day_array, bout_array, ic_seconds = day_array[:length], bout_array[:length], ic_seconds[:length]
        ranges: List[BoutRange] = []
        start_index = 0
        while start_index < length:
            end_index = start_index + 1
            while end_index < length and day_array[end_index] == day_array[start_index] and bout_array[end_index] == bout_array[start_index]:
                end_index += 1
            start_time = float(ic_seconds[start_index])
            end_time = float(ic_seconds[end_index - 1])
            if end_time <= start_time:
                end_time = start_time + 1e-6
            sample_start = int(np.searchsorted(numeric_time, start_time, side="left"))
            sample_end = int(np.searchsorted(numeric_time, end_time, side="right"))
            sample_start = max(0, min(sample_start, numeric_time.size - 1))
            sample_end = max(sample_start + 1, min(sample_end, numeric_time.size))
            ranges.append(
                BoutRange(
                    sample_start=sample_start,
                    sample_end=sample_end,
                    event_start=start_index,
                    event_end=end_index,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            start_index = end_index
        if ranges:
            return ranges
        if treadmill_profile:
            return [self._full_recording_bout(sensor_data)]
        return [self._full_recording_bout(sensor_data)]

    @staticmethod
    def _to_seconds(time_values: np.ndarray) -> np.ndarray:
        if np.issubdtype(time_values.dtype, np.datetime64):
            return time_values.astype("datetime64[ns]").astype(np.int64) / 1e9
        try:
            return time_values.astype(float)
        except Exception:
            return np.array([float(item.timestamp()) for item in time_values], dtype=float)

    @staticmethod
    def _select_accelerometer_sensor(imu_data: IMUData) -> SensorData:
        if not imu_data.data:
            raise ValueError("IMUData has no sensor streams.")
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
    def _full_recording_bout(sensor_data: SensorData) -> BoutRange:
        numeric_time = np.asarray(sensor_data.time, dtype=float)
        if numeric_time.size == 0:
            raise ValueError("Sensor time series is empty.")
        start_time, end_time = float(numeric_time[0]), float(numeric_time[-1])
        if end_time <= start_time:
            end_time = start_time + 1e-6
        return BoutRange(
            sample_start=0,
            sample_end=numeric_time.size,
            event_start=0,
            event_end=0,
            start_time=start_time,
            end_time=end_time,
        )
