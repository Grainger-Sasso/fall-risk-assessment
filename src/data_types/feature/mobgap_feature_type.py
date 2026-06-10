from enum import Enum
from typing import List


class MobgapFeatureType(Enum):
    """Per-stride features produced by the MobGap extraction backend."""

    IC_SAMPLE = "mobgap:ic_sample"
    FOOT = "mobgap:foot"
    STEP_TIME_S = "mobgap:step_time_s"
    STRIDE_TIME_S = "mobgap:stride_time_s"
    CADENCE_SPM = "mobgap:cadence_spm"
    STRIDE_LENGTH_M = "mobgap:stride_length_m"
    WALKING_SPEED_MPS = "mobgap:walking_speed_mps"

    @classmethod
    def get_stride_feature_types(cls) -> List["MobgapFeatureType"]:
        return list(cls)
