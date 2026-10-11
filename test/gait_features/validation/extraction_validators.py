from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from src.data_model.features.bout_sample_layout import (
    count_usable_samples,
    flatten_bout_features,
    usable_sample_mask,
)
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.gait_features.contracts.extraction_result import BoutSegment, GaitExtractionResult
from src.gait_features.diagnosis.stride_failure import diagnose_stride_nan_source


def assert_bout_segments_valid(
    segments: Iterable[BoutSegment],
    *,
    min_bout_duration_seconds: float = 0.0,
    min_bout_count: int = 1,
) -> None:
    segment_list = list(segments)
    if len(segment_list) < min_bout_count:
        raise AssertionError(
            f"Expected at least {min_bout_count} bout segment(s), found {len(segment_list)}"
        )
    for index, segment in enumerate(segment_list):
        if segment.end_time <= segment.start_time:
            raise AssertionError(f"Bout {index}: end_time must exceed start_time")
        if segment.sample_end <= segment.sample_start:
            raise AssertionError(f"Bout {index}: sample_end must exceed sample_start")
        duration = segment.end_time - segment.start_time
        if duration < min_bout_duration_seconds:
            raise AssertionError(
                f"Bout {index}: duration {duration:.3f}s below minimum "
                f"{min_bout_duration_seconds:.3f}s"
            )


def assert_walking_bouts_detected(segments: Iterable[BoutSegment]) -> None:
    segment_list = list(segments)
    if not segment_list:
        raise AssertionError("Expected at least one detected walking bout")
    stride_bearing = [
        segment for segment in segment_list if segment.event_end > segment.event_start
    ]
    if not stride_bearing:
        raise AssertionError(
            "Detected bout segment(s) contain no stride events "
            "(event_end <= event_start for all bouts)"
        )


def assert_epoch_window_geometry(
    record: RecordFeatures,
    *,
    window_seconds: float = 8.0,
    overlap_seconds: float = 2.0,
    tolerance_seconds: float = 0.05,
) -> None:
    epoch = record.epoch_features
    if epoch.num_samples < 1:
        raise AssertionError("Expected at least one epoch sample axis entry")

    step_seconds = window_seconds - overlap_seconds
    if epoch.sample_starts.size > 1:
        observed_step = float(epoch.sample_starts[1] - epoch.sample_starts[0])
        if abs(observed_step - step_seconds) > tolerance_seconds:
            raise AssertionError(
                f"Epoch sample step {observed_step:.3f}s != expected {step_seconds:.3f}s"
            )

    if epoch.sample_ends.size and epoch.sample_starts.size:
        observed_width = float(epoch.sample_ends[0] - epoch.sample_starts[0])
        if abs(observed_width - window_seconds) > tolerance_seconds:
            raise AssertionError(
                f"Epoch window width {observed_width:.3f}s != expected {window_seconds:.3f}s"
            )


def assert_epoch_features_reliable(record: RecordFeatures) -> None:
    epoch = record.epoch_features
    expected_types = FeatureType.get_epoch_feature_types()
    if list(epoch.feature_names) != expected_types:
        raise AssertionError("Epoch feature name ordering does not match catalog")

    if count_usable_samples(epoch.features) == 0:
        raise AssertionError("No usable epoch samples with finite values")

    mean_index = expected_types.index(FeatureType.EPOCH_VERTICAL_MEAN)
    finite_means = np.isfinite(epoch.features[:, mean_index, :])
    if not np.any(finite_means):
        raise AssertionError("Epoch vertical mean is entirely non-finite")


def assert_stride_features_reliable(
    extraction_result: GaitExtractionResult,
    record: RecordFeatures,
) -> None:
    stride = record.stride_features
    diagnosis = diagnose_stride_nan_source(
        extraction_result=extraction_result,
        bout_segments=extraction_result.bout_segments,
        stride_features=stride,
        stride_feature_names=extraction_result.stride_feature_names,
    )
    if diagnosis["stage"] != "ok":
        raise AssertionError(
            f"Stride diagnosis failed at stage '{diagnosis['stage']}': {diagnosis['detail']}"
        )

    expected_types = FeatureType.get_stride_feature_types()
    if list(stride.feature_names) != expected_types:
        raise AssertionError("Stride feature name ordering does not match catalog")

    if count_usable_samples(stride.features) == 0:
        raise AssertionError("No usable stride samples with finite values")

    stride_time_index = expected_types.index(FeatureType.STRIDE_TIME)
    stride_times = stride.features[:, stride_time_index, :]
    finite_stride_times = stride_times[np.isfinite(stride_times)]
    if finite_stride_times.size == 0:
        raise AssertionError("No finite stride time values")


def assert_record_features_schema(record: RecordFeatures) -> None:
    if record.epoch_features.sample_basis != SampleBasis.EPOCH:
        raise AssertionError("epoch_features must use SampleBasis.EPOCH")
    if record.stride_features.sample_basis != SampleBasis.STRIDE:
        raise AssertionError("stride_features must use SampleBasis.STRIDE")

    for label, bout in (
        ("epoch", record.epoch_features),
        ("stride", record.stride_features),
    ):
        if bout.features.ndim != 3:
            raise AssertionError(f"{label} features must be rank-3")
        if bout.features.shape[0] != bout.num_bouts:
            raise AssertionError(f"{label} bout axis mismatch")
        if bout.features.shape[1] != bout.num_feature_types:
            raise AssertionError(f"{label} feature axis mismatch")
        if bout.features.shape[2] != bout.num_samples:
            raise AssertionError(f"{label} sample axis mismatch")
        if len(bout.feature_names) != bout.num_feature_types:
            raise AssertionError(f"{label} feature_names length mismatch")
        if bout.usable_sample_count != count_usable_samples(bout.features):
            raise AssertionError(f"{label} usable_sample_count mismatch")

    metadata = record.feature_metadata
    if not metadata.extraction_backend:
        raise AssertionError("feature_metadata.extraction_backend must be set")
    if not metadata.extraction_profile:
        raise AssertionError("feature_metadata.extraction_profile must be set")


def assert_io_fidelity(before: RecordFeatures, after: RecordFeatures) -> None:
    if before.feature_metadata.feature_identifier.value != after.feature_metadata.feature_identifier.value:
        raise AssertionError("Feature identifier changed across I/O")
    if before.feature_metadata.imu_data_identifier.value != after.feature_metadata.imu_data_identifier.value:
        raise AssertionError("IMU identifier changed across I/O")

    for basis in (SampleBasis.EPOCH, SampleBasis.STRIDE):
        source = before.get_features_by_basis(basis)
        target = after.get_features_by_basis(basis)
        if list(source.feature_names) != list(target.feature_names):
            raise AssertionError(f"{basis.value} feature names changed across I/O")

        source_matrix = source.features
        target_matrix = target.features
        if source_matrix.shape != target_matrix.shape:
            raise AssertionError(f"{basis.value} tensor shape changed across I/O")

        source_flat = flatten_bout_features(source_matrix)
        target_flat = flatten_bout_features(target_matrix)
        mask = usable_sample_mask(source_flat)
        if not np.allclose(
            source_flat[mask],
            target_flat[mask],
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        ):
            raise AssertionError(f"{basis.value} usable sample values changed across I/O")

        if source.usable_sample_count != target.usable_sample_count:
            raise AssertionError(f"{basis.value} usable_sample_count changed across I/O")
