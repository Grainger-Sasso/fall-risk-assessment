from enum import Enum
from typing import Optional

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

    @classmethod
    def resolve(
        cls,
        backend: GaitExtractionBackendId,
        explicit_profile: Optional["ExtractionProfile"] = None,
        treadmill_profile: bool = False,
    ) -> "ExtractionProfile":
        """Pick the profile for a run: explicit profile, then treadmill flag, then backend default."""
        if explicit_profile is not None:
            return explicit_profile
        if treadmill_profile:
            return cls.TREADMILL
        if backend == GaitExtractionBackendId.MOBGAP:
            return cls.MOBGAP_HEALTHY
        return cls.FREE_LIVING
