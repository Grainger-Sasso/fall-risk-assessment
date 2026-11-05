from typing import Optional

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.record_features import RecordFeatures
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.gait_features.backends.base import GaitExtractionBackend
from src.gait_features.backends.mobgap.backend import MobgapBackend
from src.gait_features.backends.skdh.backend import SkdhBackend
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.processing.backend_requirements import validate_extraction_requirements
from src.gait_features.processing.record_feature_builder import RecordFeatureBuilder


class GaitExtractionPipeline:
    """
    Turns one IMU record into RecordFeatures. Has no knowledge of DatabaseManager.

    Flow:
      backend requirement checks -> backend extraction -> RecordFeatures build
    """

    def __init__(
        self,
        backend: GaitExtractionBackendId,
        profile: ExtractionProfile,
        extractor: GaitExtractionBackend,
        record_feature_builder: RecordFeatureBuilder,
    ):
        self._validate_compatibility(backend=backend, profile=profile)
        self.backend = backend
        self.profile = profile
        self.extractor = extractor
        self.record_feature_builder = record_feature_builder

    @classmethod
    def create(
        cls,
        backend: GaitExtractionBackendId,
        profile: ExtractionProfile,
        epoch_window_seconds: float = 8.0,
        epoch_overlap_seconds: float = 2.0,
    ) -> "GaitExtractionPipeline":
        cls._validate_compatibility(backend=backend, profile=profile)
        return cls(
            backend=backend,
            profile=profile,
            extractor=cls._build_extractor(backend=backend, profile=profile),
            record_feature_builder=RecordFeatureBuilder(
                window_seconds=epoch_window_seconds,
                overlap_seconds=epoch_overlap_seconds,
            ),
        )

    def run(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications],
    ) -> RecordFeatures:
        validate_extraction_requirements(imu_data=imu_data, backend=self.backend)
        extraction_result = self.extractor.extract(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=instrument_specifications,
            profile=self.profile,
        )
        return self.record_feature_builder.build_from_extraction(
            extraction_result=extraction_result,
            imu_data=imu_data,
            user_data=user_data,
        )

    @staticmethod
    def _validate_compatibility(
        backend: GaitExtractionBackendId,
        profile: ExtractionProfile,
    ) -> None:
        if profile.backend != backend:
            raise ValueError(
                f"Extraction profile '{profile.value}' is not compatible with backend "
                f"'{backend.value}'."
            )

    @staticmethod
    def _build_extractor(
        backend: GaitExtractionBackendId,
        profile: ExtractionProfile,
    ) -> GaitExtractionBackend:
        if backend == GaitExtractionBackendId.SKDH:
            return SkdhBackend(profile=profile)
        if backend == GaitExtractionBackendId.MOBGAP:
            return MobgapBackend(profile=profile)
        raise ValueError(f"Unsupported extraction backend: {backend.value}")
