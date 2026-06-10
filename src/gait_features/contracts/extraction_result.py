from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.data_types.feature.stride_feature_name import StrideFeatureName
from src.gait_features.contracts.provenance import ExtractionProvenance


@dataclass
class BoutSegment:
    sample_start: int
    sample_end: int
    event_start: int
    event_end: int
    start_time: float
    end_time: float


@dataclass
class GaitExtractionResult:
    provenance: ExtractionProvenance
    bout_segments: List[BoutSegment]
    stride_feature_names: List[StrideFeatureName]
    stride_feature_series: Dict[StrideFeatureName, List[np.ndarray]]
    raw_debug: Optional[Dict[str, Any]] = field(default=None)
