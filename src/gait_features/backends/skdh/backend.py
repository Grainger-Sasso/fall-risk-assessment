from typing import Any, Dict, Optional

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.gait_features.backends.skdh.bout_resolver import resolve_skdh_bout_segments
from src.gait_features.backends.skdh.gait_results import GaitResults
from src.gait_features.backends.skdh.pipeline import SkdhPipelineBuilder
from src.gait_features.backends.skdh.stride_mapper import SkdhStrideMapper
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.contracts.extraction_result import GaitExtractionResult
from src.gait_features.contracts.provenance import ExtractionProvenance
from src.gait_features.processing.imu_validation import ImuValidationMixin


class SkdhBackend(ImuValidationMixin):
    def __init__(self, profile: ExtractionProfile = ExtractionProfile.FREE_LIVING):
        self.profile = profile
        self.treadmill_profile = profile == ExtractionProfile.TREADMILL
        self._pipeline_builder = SkdhPipelineBuilder()
        self._stride_mapper = SkdhStrideMapper()
        self.pipeline = self._pipeline_builder.build_pipeline(profile)
        self.gait_res_key = SkdhPipelineBuilder.GAIT_RES_KEY

    def extract(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications] = None,
        profile: Optional[ExtractionProfile] = None,
    ) -> GaitExtractionResult:
        active_profile = profile or self.profile
        gait_data = self._run_skdh_pipeline(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=instrument_specifications,
            profile=active_profile,
        )
        sensor_data = self.select_triaxial_accelerometer_sensor(imu_data)
        bout_segments = resolve_skdh_bout_segments(
            gait_data=gait_data,
            sensor_data=sensor_data,
            treadmill_profile=active_profile == ExtractionProfile.TREADMILL,
        )
        bout_event_ranges = [(item.event_start, item.event_end) for item in bout_segments]
        stride_feature_names, stride_feature_series = self._stride_mapper.map_to_series(
            gait_data=gait_data,
            bout_event_ranges=bout_event_ranges,
        )
        provenance = ExtractionProvenance.create(
            backend=GaitExtractionBackendId.SKDH.value,
            stride_feature_catalog=GaitExtractionBackendId.SKDH.value,
            profile=active_profile.value,
            library_version=SkdhPipelineBuilder.library_version(),
        )
        return GaitExtractionResult(
            provenance=provenance,
            bout_segments=bout_segments,
            stride_feature_names=stride_feature_names,
            stride_feature_series=stride_feature_series,
            raw_debug={"gait_data": gait_data},
        )

    def _run_skdh_pipeline(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications],
        profile: ExtractionProfile,
    ) -> Dict[str, Any]:
        if self.pipeline is None:
            raise ModuleNotFoundError(
                "scikit-digital-health (skdh) is required for gait feature extraction."
            )
        treadmill_profile = profile == ExtractionProfile.TREADMILL
        sensor_data, accel, time = self.prepare_accelerometer_for_backend(
            imu_data=imu_data,
            instrument_specifications=instrument_specifications,
            treadmill_profile=treadmill_profile,
            backend=GaitExtractionBackendId.SKDH,
        )
        height = self.normalize_height(user_data.clinical_demographic_data.height.value)
        result = self.pipeline.run(time=time, accel=accel, height=height)
        if self.gait_res_key not in result:
            raise KeyError(
                f"SKDH pipeline output missing expected key '{self.gait_res_key}'."
            )
        return result[self.gait_res_key]


class GaitFeatureExtractor(SkdhBackend):
    """Backward-compatible SKDH extractor."""

    def __init__(self, treadmill_profile: bool = False):
        profile = ExtractionProfile.from_treadmill_flag(treadmill_profile)
        super().__init__(profile=profile)

    def extract_gait_features(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications] = None,
    ) -> GaitResults:
        gait_data = self._run_skdh_pipeline(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=instrument_specifications,
            profile=self.profile,
        )
        return GaitResults(gait_data)
