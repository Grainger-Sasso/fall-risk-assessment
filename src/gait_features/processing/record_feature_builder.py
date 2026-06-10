from typing import List, Optional, Tuple

import numpy as np

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_units import get_stride_feature_unit
from src.data_types.feature.stride_feature_name import StrideFeatureName
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.gait_features.backends.skdh.bout_resolver import resolve_skdh_bout_segments
from src.gait_features.backends.skdh.gait_results import GaitResults
from src.gait_features.backends.skdh.stride_mapper import SkdhStrideMapper
from src.gait_features.contracts.extraction_result import BoutSegment, GaitExtractionResult
from src.gait_features.diagnosis.stride_failure import (
    StrideFeatureGenerationError,
    diagnose_stride_nan_source,
)
from src.gait_features.processing.epoch_feature_generator import EpochFeatureGenerator
from src.gait_features.processing.imu_validation import ImuValidationMixin
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.identifier import IdentifierGenerator


class RecordFeatureBuilder(ImuValidationMixin):
    def __init__(
        self,
        window_seconds: float = 8.0,
        overlap_seconds: float = 2.0,
        feature_id_generator: Optional[IdentifierGenerator[FeatureIdentifier]] = None,
    ):
        self.feature_id_generator = feature_id_generator or IdentifierGenerator(
            "feature", FeatureIdentifier
        )
        self.skdh_stride_mapper = SkdhStrideMapper()
        self.epoch_generator = EpochFeatureGenerator(
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        )

    def build_from_extraction(
        self,
        extraction_result: GaitExtractionResult,
        imu_data: IMUData,
        user_data: UserData,
    ) -> RecordFeatures:
        sensor_data = self.select_triaxial_accelerometer_sensor(imu_data)
        stride_features = self._build_stride_features(extraction_result)
        self._validate_stride_feature_generation(
            extraction_result=extraction_result,
            bout_segments=extraction_result.bout_segments,
            stride_features=stride_features,
            stride_feature_names=extraction_result.stride_feature_names,
        )
        accel_mps2 = self.prepare_accel_matrix_mps2(sensor_data)
        epoch_features = self.epoch_generator.build(
            sensor_data=sensor_data,
            bout_sample_ranges=[
                (segment.sample_start, segment.sample_end)
                for segment in extraction_result.bout_segments
            ],
            bout_time_ranges=[
                (segment.start_time, segment.end_time)
                for segment in extraction_result.bout_segments
            ],
            accel_mps2=accel_mps2,
        )
        provenance = extraction_result.provenance
        metadata = FeatureMetadata(
            feature_identifier=self.feature_id_generator.generate_identifier(),
            user_identifier=user_data.user_identifier,
            imu_data_identifier=imu_data.metadata.imu_data_identifier,
            extraction_backend=provenance.backend,
            stride_feature_catalog=provenance.stride_feature_catalog,
            extraction_profile=provenance.profile,
            extraction_library_version=provenance.library_version,
            extracted_at_utc=provenance.extracted_at_utc,
        )
        return RecordFeatures(
            epoch_features=epoch_features,
            stride_features=stride_features,
            feature_metadata=metadata,
        )

    def _build_stride_features(self, extraction_result: GaitExtractionResult) -> BoutFeatures:
        feature_names = extraction_result.stride_feature_names
        bout_segments = extraction_result.bout_segments
        num_bouts = len(bout_segments)
        num_features = len(feature_names)
        max_samples = self._resolve_max_samples(extraction_result.stride_feature_series, num_bouts)
        features = np.full((num_bouts, num_features, max_samples), np.nan, dtype=float)

        for bout_index in range(num_bouts):
            for feature_index, feature_name in enumerate(feature_names):
                values_list = extraction_result.stride_feature_series.get(feature_name, [])
                if bout_index >= len(values_list):
                    continue
                values = np.asarray(values_list[bout_index], dtype=float)
                if values.size > 0:
                    features[bout_index, feature_index, : values.size] = values

        if max_samples == 0:
            sample_starts = np.array([], dtype=float)
            sample_ends = np.array([], dtype=float)
        else:
            sample_starts = np.arange(max_samples, dtype=float)
            sample_ends = sample_starts + 1.0

        return BoutFeatures(
            sample_basis=SampleBasis.STRIDE,
            features=features,
            bout_starts=np.array([segment.start_time for segment in bout_segments], dtype=float),
            bout_ends=np.array([segment.end_time for segment in bout_segments], dtype=float),
            feature_names=feature_names,
            sample_starts=sample_starts,
            sample_ends=sample_ends,
            units=[get_stride_feature_unit(feature_name) for feature_name in feature_names],
        )

    @staticmethod
    def _resolve_max_samples(
        stride_feature_series: dict,
        num_bouts: int,
    ) -> int:
        max_samples = 0
        for values_list in stride_feature_series.values():
            for bout_index in range(num_bouts):
                if bout_index >= len(values_list):
                    continue
                max_samples = max(max_samples, int(np.asarray(values_list[bout_index]).size))
        if num_bouts == 0:
            return 0
        return max(1, max_samples)

    def _validate_stride_feature_generation(
        self,
        extraction_result: GaitExtractionResult,
        bout_segments: List[BoutSegment],
        stride_features: BoutFeatures,
        stride_feature_names: List[StrideFeatureName],
    ) -> None:
        diagnosis = diagnose_stride_nan_source(
            extraction_result=extraction_result,
            bout_segments=bout_segments,
            stride_features=stride_features,
            stride_feature_names=stride_feature_names,
        )
        if diagnosis["stage"] != "ok":
            raise StrideFeatureGenerationError(
                f"Stride feature generation failed [{diagnosis['stage']}]: "
                f"{diagnosis['detail']}",
                diagnosis=diagnosis,
            )


class RecordFeatureGenerationBuilder(RecordFeatureBuilder):
    """Backward-compatible builder supporting legacy GaitResults input."""

    def build(
        self,
        gait_results: GaitResults,
        imu_data: IMUData,
        user_data: UserData,
        treadmill_profile: bool = False,
    ) -> RecordFeatures:
        sensor_data = self.select_triaxial_accelerometer_sensor(imu_data)
        bout_segments = resolve_skdh_bout_ranges_legacy(
            gait_results=gait_results,
            sensor_data=sensor_data,
            treadmill_profile=treadmill_profile,
        )
        stride_features = self.skdh_stride_mapper.build(
            gait_results=gait_results,
            bout_event_ranges=[(item.event_start, item.event_end) for item in bout_segments],
            bout_time_ranges=[(item.start_time, item.end_time) for item in bout_segments],
        )
        extraction_result = GaitExtractionResult(
            provenance=extraction_result_provenance_for_legacy_skdh(treadmill_profile),
            bout_segments=bout_segments,
            stride_feature_names=self.skdh_stride_mapper.feature_types,
            stride_feature_series={
                feature_type: [
                    np.asarray(
                        self.skdh_stride_mapper._extract_feature_slice(
                            gait_data=gait_results.data,
                            feature_type=feature_type,
                            start_index=event_start,
                            end_index=event_end,
                            fallback_length=max(1, event_end - event_start),
                        ),
                        dtype=float,
                    )
                    for event_start, event_end in [
                        (segment.event_start, segment.event_end) for segment in bout_segments
                    ]
                ]
                for feature_type in self.skdh_stride_mapper.feature_types
            },
            raw_debug={"gait_data": gait_results.data},
        )
        self._validate_stride_feature_generation(
            extraction_result=extraction_result,
            bout_segments=bout_segments,
            stride_features=stride_features,
            stride_feature_names=self.skdh_stride_mapper.feature_types,
        )
        accel_mps2 = self.prepare_accel_matrix_mps2(sensor_data)
        epoch_features = self.epoch_generator.build(
            sensor_data=sensor_data,
            bout_sample_ranges=[
                (segment.sample_start, segment.sample_end) for segment in bout_segments
            ],
            bout_time_ranges=[
                (segment.start_time, segment.end_time) for segment in bout_segments
            ],
            accel_mps2=accel_mps2,
        )
        provenance = extraction_result.provenance
        metadata = FeatureMetadata(
            feature_identifier=self.feature_id_generator.generate_identifier(),
            user_identifier=user_data.user_identifier,
            imu_data_identifier=imu_data.metadata.imu_data_identifier,
            extraction_backend=provenance.backend,
            stride_feature_catalog=provenance.stride_feature_catalog,
            extraction_profile=provenance.profile,
            extraction_library_version=provenance.library_version,
            extracted_at_utc=provenance.extracted_at_utc,
        )
        return RecordFeatures(
            epoch_features=epoch_features,
            stride_features=stride_features,
            feature_metadata=metadata,
        )


def resolve_skdh_bout_ranges_legacy(
    gait_results: GaitResults,
    sensor_data: SensorData,
    treadmill_profile: bool,
) -> List[BoutSegment]:
    return resolve_skdh_bout_segments(
        gait_data=gait_results.data,
        sensor_data=sensor_data,
        treadmill_profile=treadmill_profile,
    )


def extraction_result_provenance_for_legacy_skdh(treadmill_profile: bool):
    from src.gait_features.backends.skdh.pipeline import SkdhPipelineBuilder
    from src.gait_features.config.extraction_backend import GaitExtractionBackendId
    from src.gait_features.config.extraction_profile import ExtractionProfile
    from src.gait_features.contracts.provenance import ExtractionProvenance

    profile = (
        ExtractionProfile.TREADMILL if treadmill_profile else ExtractionProfile.FREE_LIVING
    )
    return ExtractionProvenance.create(
        backend=GaitExtractionBackendId.SKDH.value,
        stride_feature_catalog=GaitExtractionBackendId.SKDH.value,
        profile=profile.value,
        library_version=SkdhPipelineBuilder.library_version(),
    )
