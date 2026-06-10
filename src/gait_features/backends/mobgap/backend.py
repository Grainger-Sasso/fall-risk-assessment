from typing import Optional

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.gait_features.backends.mobgap.bout_resolver import resolve_mobgap_bout_segments
from src.gait_features.backends.mobgap.imu_adapter import MobgapImuAdapter
from src.gait_features.backends.mobgap.stride_mapper import MobgapStrideMapper
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.processing.backend_requirements import validate_extraction_requirements
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.contracts.extraction_result import GaitExtractionResult
from src.gait_features.contracts.provenance import ExtractionProvenance

try:
    from mobgap.pipeline import MobilisedPipelineHealthy
except ModuleNotFoundError:
    MobilisedPipelineHealthy = None


class MobgapBackend:
    def __init__(self, profile: ExtractionProfile = ExtractionProfile.MOBGAP_HEALTHY):
        if profile != ExtractionProfile.MOBGAP_HEALTHY:
            raise ValueError(
                f"MobGap backend does not support extraction profile '{profile.value}'."
            )
        self.profile = profile
        self._adapter = MobgapImuAdapter()
        self._stride_mapper = MobgapStrideMapper()

    def extract(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications] = None,
        profile: Optional[ExtractionProfile] = None,
    ) -> GaitExtractionResult:
        active_profile = profile or self.profile
        if active_profile != ExtractionProfile.MOBGAP_HEALTHY:
            raise ValueError(
                f"MobGap backend does not support extraction profile '{active_profile.value}'."
            )
        validate_extraction_requirements(
            imu_data=imu_data,
            backend=GaitExtractionBackendId.MOBGAP,
        )
        if MobilisedPipelineHealthy is None:
            raise ModuleNotFoundError(
                "mobgap is required for MobGap gait feature extraction."
            )

        datapoint, sensor_data, sampling_rate_hz = self._adapter.build_datapoint(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=instrument_specifications,
        )
        pipeline = MobilisedPipelineHealthy().run(datapoint)
        per_wb_parameters = pipeline.per_wb_parameters_
        per_stride_parameters = pipeline.per_stride_parameters_
        raw_ic_list = pipeline.raw_ic_list_
        bout_segments = resolve_mobgap_bout_segments(
            per_wb_parameters=per_wb_parameters,
            sensor_data=sensor_data,
        )
        stride_feature_names, stride_feature_series = self._stride_mapper.map_to_series(
            per_stride_parameters=per_stride_parameters,
            raw_ic_list=raw_ic_list,
            bout_segments=bout_segments,
            sampling_rate_hz=sampling_rate_hz,
        )
        provenance = ExtractionProvenance.create(
            backend=GaitExtractionBackendId.MOBGAP.value,
            stride_feature_catalog=GaitExtractionBackendId.MOBGAP.value,
            profile=active_profile.value,
            library_version=self._library_version(),
        )
        return GaitExtractionResult(
            provenance=provenance,
            bout_segments=bout_segments,
            stride_feature_names=stride_feature_names,
            stride_feature_series=stride_feature_series,
            raw_debug={
                "per_wb_parameters": per_wb_parameters,
                "per_stride_parameters": per_stride_parameters,
                "raw_ic_list": raw_ic_list,
                "sampling_rate_hz": sampling_rate_hz,
            },
        )

    @staticmethod
    def _library_version() -> str:
        try:
            import mobgap

            return getattr(mobgap, "__version__", "unknown")
        except ModuleNotFoundError:
            return "unavailable"
