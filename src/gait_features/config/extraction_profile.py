from enum import Enum

from src.gait_features.config.extraction_backend import GaitExtractionBackendId


class ExtractionProfile(Enum):
    FREE_LIVING = "free_living"
    TREADMILL = "treadmill"
    MOBGAP_HEALTHY = "mobgap_healthy"

    @property
    def backend(self) -> GaitExtractionBackendId:
        if self == ExtractionProfile.MOBGAP_HEALTHY:
            return GaitExtractionBackendId.MOBGAP
        return GaitExtractionBackendId.SKDH

    @classmethod
    def from_treadmill_flag(cls, treadmill_profile: bool) -> "ExtractionProfile":
        return cls.TREADMILL if treadmill_profile else cls.FREE_LIVING
