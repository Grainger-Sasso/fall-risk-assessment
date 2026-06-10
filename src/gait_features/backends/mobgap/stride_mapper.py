from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_model.features.bout_features import BoutFeatures
from src.data_types.feature.mobgap_feature_type import MobgapFeatureType
from src.data_types.feature.feature_units import get_stride_feature_unit
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.gait_features.contracts.extraction_result import BoutSegment, StrideFeatureName


class MobgapStrideMapper:
    def __init__(self, feature_types: Optional[List[MobgapFeatureType]] = None):
        self.feature_types = feature_types or MobgapFeatureType.get_stride_feature_types()

    def map_to_series(
        self,
        per_stride_parameters: pd.DataFrame,
        raw_ic_list: pd.DataFrame,
        bout_segments: List[BoutSegment],
        sampling_rate_hz: float,
    ) -> Tuple[List[StrideFeatureName], Dict[StrideFeatureName, List[np.ndarray]]]:
        series: Dict[StrideFeatureName, List[np.ndarray]] = {
            feature_type: [] for feature_type in self.feature_types
        }
        step_times_by_wb = self._compute_step_times_by_wb(
            raw_ic_list, bout_segments, sampling_rate_hz
        )

        for bout_index, segment in enumerate(bout_segments):
            bout_strides = self._select_strides_for_bout(per_stride_parameters, segment)
            bout_arrays = self._build_stride_arrays(
                bout_strides=bout_strides,
                step_times=step_times_by_wb.get(bout_index, np.array([], dtype=float)),
            )
            for feature_type in self.feature_types:
                series[feature_type].append(bout_arrays[feature_type])
        return list(self.feature_types), series

    def build_bout_features(
        self,
        per_stride_parameters: pd.DataFrame,
        raw_ic_list: pd.DataFrame,
        bout_segments: List[BoutSegment],
        sampling_rate_hz: float,
    ) -> BoutFeatures:
        feature_names, series = self.map_to_series(
            per_stride_parameters=per_stride_parameters,
            raw_ic_list=raw_ic_list,
            bout_segments=bout_segments,
            sampling_rate_hz=sampling_rate_hz,
        )
        num_bouts = len(bout_segments)
        num_features = len(feature_names)
        max_samples = max(
            (values.size for values_list in series.values() for values in values_list),
            default=0,
        )
        max_samples = max(1, max_samples) if num_bouts > 0 else 0
        features = np.full((num_bouts, num_features, max_samples), np.nan, dtype=float)

        for bout_index in range(num_bouts):
            for feature_index, feature_name in enumerate(feature_names):
                values = series[feature_name][bout_index]
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
            bout_starts=np.array([segment.start_time for segment in bout_segments], dtype=float),
            bout_ends=np.array([segment.end_time for segment in bout_segments], dtype=float),
            feature_names=feature_names,
            sample_starts=sample_starts,
            sample_ends=sample_ends,
            units=[get_stride_feature_unit(feature_name) for feature_name in feature_names],
        )

    def _select_strides_for_bout(
        self,
        per_stride_parameters: pd.DataFrame,
        segment: BoutSegment,
    ) -> pd.DataFrame:
        if per_stride_parameters is None or per_stride_parameters.empty:
            return pd.DataFrame()
        if "wb_id" in per_stride_parameters.index.names:
            try:
                return per_stride_parameters.xs(segment.event_start, level="wb_id")
            except (KeyError, ValueError):
                pass
        mask = (per_stride_parameters["start"] >= segment.sample_start) & (
            per_stride_parameters["start"] < segment.sample_end
        )
        return per_stride_parameters.loc[mask]

    def _build_stride_arrays(
        self,
        bout_strides: pd.DataFrame,
        step_times: np.ndarray,
    ) -> Dict[MobgapFeatureType, np.ndarray]:
        count = len(bout_strides)
        arrays = {feature_type: np.full(count, np.nan, dtype=float) for feature_type in self.feature_types}
        if count == 0:
            return arrays

        for index, (_, row) in enumerate(bout_strides.iterrows()):
            arrays[MobgapFeatureType.IC_SAMPLE][index] = float(row.get("start", np.nan))
            arrays[MobgapFeatureType.FOOT][index] = self._encode_foot(row.get("lr_label"))
            arrays[MobgapFeatureType.STRIDE_TIME_S][index] = float(row.get("stride_duration_s", np.nan))
            arrays[MobgapFeatureType.CADENCE_SPM][index] = float(row.get("cadence_spm", np.nan))
            arrays[MobgapFeatureType.STRIDE_LENGTH_M][index] = float(row.get("stride_length_m", np.nan))
            arrays[MobgapFeatureType.WALKING_SPEED_MPS][index] = float(row.get("walking_speed_mps", np.nan))
            if index < step_times.size:
                arrays[MobgapFeatureType.STEP_TIME_S][index] = float(step_times[index])
        return arrays

    def _compute_step_times_by_wb(
        self,
        raw_ic_list: pd.DataFrame,
        bout_segments: List[BoutSegment],
        sampling_rate_hz: float,
    ) -> Dict[int, np.ndarray]:
        if raw_ic_list is None or raw_ic_list.empty or sampling_rate_hz <= 0:
            return {}
        step_times_by_wb: Dict[int, np.ndarray] = {}
        ics = np.asarray(raw_ic_list["ic"], dtype=float)
        for bout_index, segment in enumerate(bout_segments):
            mask = (ics >= segment.sample_start) & (ics < segment.sample_end)
            bout_ics = ics[mask]
            if bout_ics.size < 2:
                step_times_by_wb[bout_index] = np.array([], dtype=float)
                continue
            step_times_by_wb[bout_index] = np.diff(bout_ics) / sampling_rate_hz
        return step_times_by_wb

    @staticmethod
    def _encode_foot(label: Any) -> float:
        if label is None or (isinstance(label, float) and np.isnan(label)):
            return np.nan
        normalized = str(label).strip().lower()
        if normalized == "right":
            return 1.0
        if normalized == "left":
            return 0.0
        return np.nan
