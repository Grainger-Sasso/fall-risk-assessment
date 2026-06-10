from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.features.bout_features import BoutFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.feature.feature_units import get_feature_unit
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


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
        accel_mps2: Optional[np.ndarray] = None,
    ) -> BoutFeatures:
        if accel_mps2 is not None:
            if accel_mps2.shape[0] != len(sensor_data.time):
                raise ValueError(
                    "accel_mps2 row count must match sensor time length "
                    f"({accel_mps2.shape[0]} vs {len(sensor_data.time)})."
                )
            x = np.asarray(accel_mps2[:, 0], dtype=float)
            y = np.asarray(accel_mps2[:, 1], dtype=float)
            z = np.asarray(accel_mps2[:, 2], dtype=float)
        else:
            x = np.asarray(
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data,
                dtype=float,
            )
            y = np.asarray(
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data,
                dtype=float,
            )
            z = np.asarray(
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data,
                dtype=float,
            )
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
