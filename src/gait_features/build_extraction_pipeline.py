from typing import Tuple

from src.gait_features.backends.base import GaitExtractionBackend
from src.gait_features.backends.mobgap.backend import MobgapBackend
from src.gait_features.backends.skdh.backend import SkdhBackend
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.processing.record_feature_builder import RecordFeatureBuilder


def build_extraction_pipeline(
    backend: GaitExtractionBackendId,
    profile: ExtractionProfile,
    epoch_window_seconds: float = 8.0,
    epoch_overlap_seconds: float = 2.0,
) -> Tuple[GaitExtractionBackend, RecordFeatureBuilder]:
    if profile.backend != backend:
        raise ValueError(
            f"Extraction profile '{profile.value}' is not compatible with backend "
            f"'{backend.value}'."
        )
    if backend == GaitExtractionBackendId.SKDH:
        extractor: GaitExtractionBackend = SkdhBackend(profile=profile)
    elif backend == GaitExtractionBackendId.MOBGAP:
        extractor = MobgapBackend(profile=profile)
    else:
        raise ValueError(f"Unsupported extraction backend: {backend.value}")

    record_builder = RecordFeatureBuilder(
        window_seconds=epoch_window_seconds,
        overlap_seconds=epoch_overlap_seconds,
    )
    return extractor, record_builder
