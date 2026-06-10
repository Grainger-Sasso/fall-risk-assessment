from typing import Any, Dict, List

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_types.feature.stride_feature_name import StrideFeatureName, stride_feature_name_value
from src.gait_features.backends.skdh.gait_results import GaitResults
from src.gait_features.backends.skdh.pipeline import SkdhPipelineBuilder
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.contracts.extraction_result import BoutSegment, GaitExtractionResult
from src.gait_features.contracts.provenance import ExtractionProvenance


class StrideFeatureGenerationError(ValueError):
    def __init__(self, message: str, diagnosis: Dict[str, Any]):
        super().__init__(message)
        self.diagnosis = diagnosis


def diagnose_stride_nan_source(
    extraction_result: GaitExtractionResult,
    bout_segments: List[BoutSegment],
    stride_features: BoutFeatures,
    stride_feature_names: List[StrideFeatureName],
) -> Dict[str, Any]:
    feature_values = np.asarray(stride_features.features, dtype=float)
    gait_data = extraction_result.raw_debug.get("gait_data", {}) if extraction_result.raw_debug else {}
    present_keys = [
        stride_feature_name_value(name)
        for name in stride_feature_names
        if stride_feature_name_value(name) in gait_data
        or name in extraction_result.stride_feature_series
    ]
    missing_keys = [
        stride_feature_name_value(name)
        for name in stride_feature_names
        if stride_feature_name_value(name) not in present_keys
    ]
    bouts_without_events = [
        idx
        for idx, item in enumerate(bout_segments)
        if item.event_end <= item.event_start
    ]
    if feature_values.size == 0:
        bouts_all_nan: List[int] = []
        tensor_all_nan = True
    else:
        tensor_all_nan = bool(np.isnan(feature_values).all())
        bouts_all_nan = [
            idx for idx in range(feature_values.shape[0]) if np.isnan(feature_values[idx]).all()
        ]

    report: Dict[str, Any] = {
        "stage": "ok",
        "tensor_shape": tuple(int(dim) for dim in feature_values.shape),
        "expected_stride_keys": len(stride_feature_names),
        "present_stride_keys": present_keys,
        "missing_stride_keys": missing_keys,
        "bouts_without_events": bouts_without_events,
        "tensor_all_nan": tensor_all_nan,
        "bouts_all_nan": bouts_all_nan,
        "extraction_backend": extraction_result.provenance.backend,
        "detail": "Stride values are present.",
    }

    if feature_values.size == 0:
        report["stage"] = "stride_tensor_all_nan"
        report["detail"] = "Stride tensor is empty."
        return report
    if len(present_keys) == 0 and not any(
        any(np.isfinite(np.asarray(values)).any() if values.size else False)
        for values_list in extraction_result.stride_feature_series.values()
        for values in values_list
    ):
        report["stage"] = "missing_stride_features"
        report["detail"] = (
            "Extraction output contained none of the expected stride feature keys; "
            "no gait bouts/strides were detected (likely wrong extraction profile)."
        )
        return report
    if bouts_without_events:
        report["stage"] = "no_stride_events"
        report["detail"] = (
            "Zero-width stride event range(s) for bout index/indices "
            f"{bouts_without_events}; no stride events were detected to slice."
        )
        return report
    if tensor_all_nan:
        report["stage"] = "stride_tensor_all_nan"
        report["detail"] = "All stride feature values are NaN."
        return report
    if bouts_all_nan:
        report["stage"] = "stride_tensor_all_nan"
        report["detail"] = (
            f"NaN-only stride values for bout index/indices {bouts_all_nan}."
        )
        return report
    return report


def diagnose_stride_nan_source_legacy(
    gait_results: GaitResults,
    bout_ranges: List[BoutSegment],
    stride_features: BoutFeatures,
    stride_feature_types: List[StrideFeatureName],
) -> Dict[str, Any]:
    extraction_result = GaitExtractionResult(
        provenance=ExtractionProvenance.create(
            backend=GaitExtractionBackendId.SKDH.value,
            stride_feature_catalog=GaitExtractionBackendId.SKDH.value,
            profile=ExtractionProfile.FREE_LIVING.value,
            library_version=SkdhPipelineBuilder.library_version(),
        ),
        bout_segments=bout_ranges,
        stride_feature_names=stride_feature_types,
        stride_feature_series={},
        raw_debug={"gait_data": gait_results.data},
    )
    return diagnose_stride_nan_source(
        extraction_result=extraction_result,
        bout_segments=bout_ranges,
        stride_features=stride_features,
        stride_feature_names=stride_feature_types,
    )
