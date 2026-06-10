"""Backward-compatible re-exports for gait feature extraction."""

from src.gait_features.backends.skdh.backend import GaitFeatureExtractor, SkdhBackend
from src.gait_features.backends.skdh.gait_results import GaitResults
from src.gait_features.backends.skdh.stride_mapper import SkdhStrideMapper
from src.gait_features.contracts.extraction_result import BoutSegment
from src.gait_features.diagnosis.stride_failure import (
    StrideFeatureGenerationError,
    diagnose_stride_nan_source_legacy as diagnose_stride_nan_source,
)
from src.gait_features.processing.epoch_feature_generator import EpochFeatureGenerator
from src.gait_features.processing.record_feature_builder import RecordFeatureGenerationBuilder

# Legacy aliases
BoutRange = BoutSegment
StrideFeatureMapper = SkdhStrideMapper

__all__ = [
    "BoutRange",
    "BoutSegment",
    "EpochFeatureGenerator",
    "GaitFeatureExtractor",
    "GaitResults",
    "RecordFeatureGenerationBuilder",
    "SkdhBackend",
    "StrideFeatureGenerationError",
    "StrideFeatureMapper",
    "diagnose_stride_nan_source",
]
