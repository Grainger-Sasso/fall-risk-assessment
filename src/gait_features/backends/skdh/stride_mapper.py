from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.feature.feature_units import get_stride_feature_unit
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.gait_features.backends.skdh.gait_results import GaitResults
from src.gait_features.contracts.extraction_result import StrideFeatureName


class SkdhStrideMapper:
    def __init__(self, feature_types: Optional[List[FeatureType]] = None):
        self.feature_types = feature_types or FeatureType.get_stride_feature_types()

    def build(
        self,
        gait_results: GaitResults,
        bout_event_ranges: List[Tuple[int, int]],
        bout_time_ranges: List[Tuple[float, float]],
    ) -> BoutFeatures:
        _, series = self.map_to_series(
            gait_data=gait_results.data,
            bout_event_ranges=bout_event_ranges,
        )
        return self._series_to_bout_features(series, bout_event_ranges, bout_time_ranges)

    def map_to_series(
        self,
        gait_data: Dict[str, Any],
        bout_event_ranges: List[Tuple[int, int]],
    ) -> Tuple[List[StrideFeatureName], Dict[StrideFeatureName, List[np.ndarray]]]:
        series: Dict[StrideFeatureName, List[np.ndarray]] = {
            feature_type: [] for feature_type in self.feature_types
        }
        for event_start, event_end in bout_event_ranges:
            max_samples = max(1, max(0, event_end - event_start)) if bout_event_ranges else 0
            for feature_type in self.feature_types:
                values = self._extract_feature_slice(
                    gait_data=gait_data,
                    feature_type=feature_type,
                    start_index=event_start,
                    end_index=event_end,
                    fallback_length=max_samples,
                )
                series[feature_type].append(values)
        return list(self.feature_types), series

    def _extract_feature_slice(
        self,
        gait_data: Dict[str, Any],
        feature_type: FeatureType,
        start_index: int,
        end_index: int,
        fallback_length: int,
    ) -> np.ndarray:
        raw_values = gait_data.get(feature_type.value)
        if raw_values is None:
            return (
                np.array([np.nan], dtype=float)
                if start_index == end_index and fallback_length > 0
                else np.array([], dtype=float)
            )
        values = np.asarray(raw_values, dtype=float)
        if values.size == 0:
            return (
                np.array([np.nan], dtype=float)
                if start_index == end_index and fallback_length > 0
                else np.array([], dtype=float)
            )
        bounded_start = max(0, min(start_index, values.size))
        bounded_end = max(bounded_start, min(end_index, values.size))
        sliced = values[bounded_start:bounded_end]
        if sliced.size == 0 and fallback_length > 0:
            return np.array([np.nan], dtype=float)
        return sliced

    def _series_to_bout_features(
        self,
        series: Dict[StrideFeatureName, List[np.ndarray]],
        bout_event_ranges: List[Tuple[int, int]],
        bout_time_ranges: List[Tuple[float, float]],
    ) -> BoutFeatures:
        num_bouts = len(bout_event_ranges)
        num_features = len(self.feature_types)
        max_samples = self._resolve_max_samples(bout_event_ranges)
        features = np.full((num_bouts, num_features, max_samples), np.nan, dtype=float)

        for bout_index, _ in enumerate(bout_event_ranges):
            for feature_index, feature_type in enumerate(self.feature_types):
                values = series.get(feature_type, [np.array([], dtype=float)])[bout_index]
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
            units=[get_stride_feature_unit(feature_type) for feature_type in self.feature_types],
        )

    @staticmethod
    def _resolve_max_samples(bout_event_ranges: List[Tuple[int, int]]) -> int:
        if not bout_event_ranges:
            return 0
        return max(1, max(0, max(end - start for start, end in bout_event_ranges)))
