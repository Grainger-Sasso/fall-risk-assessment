from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data_model.features.bout_sample_layout import flatten_bout_features, usable_sample_mask
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from test.fixtures.clinical_reference import (
    default_gait_speed_tolerance_relative,
    lab_walks_base_velocity,
)


@dataclass(frozen=True)
class GaitSpeedReferenceAudit:
    """Non-throwing gait-speed reference comparison outcome for reporting."""

    participant_id: str
    passed: bool
    extracted_mps: Optional[float] = None
    reference_mps: Optional[float] = None
    relative_error: Optional[float] = None
    tolerance_relative: Optional[float] = None
    error: Optional[str] = None


def median_stride_gait_speed_mps(record: RecordFeatures) -> float:
    """Median finite GAIT_SPEED over usable stride samples."""
    stride = record.stride_features
    if FeatureType.GAIT_SPEED not in stride.feature_names:
        raise AssertionError("stride_features missing GAIT_SPEED")

    speed_index = list(stride.feature_names).index(FeatureType.GAIT_SPEED)
    per_sample = flatten_bout_features(stride.features)
    mask = usable_sample_mask(per_sample)
    speeds = per_sample[mask, speed_index]
    finite_speeds = speeds[np.isfinite(speeds)]
    if finite_speeds.size == 0:
        raise AssertionError("No finite GAIT_SPEED values on usable stride samples")
    return float(np.median(finite_speeds))


def audit_gait_speed_matches_reference(
    record: RecordFeatures,
    participant_id: str,
    *,
    reference_speed_mps: Optional[float] = None,
    tolerance_relative: Optional[float] = None,
) -> GaitSpeedReferenceAudit:
    """Compare median extracted stride GAIT_SPEED to clinical reference without raising."""
    reference = reference_speed_mps
    if reference is None:
        reference = lab_walks_base_velocity(participant_id)
    if reference is None:
        return GaitSpeedReferenceAudit(
            participant_id=participant_id,
            passed=False,
            error=f"No lab-walk base(velocity) reference configured for participant {participant_id}",
        )
    if reference <= 0.0:
        return GaitSpeedReferenceAudit(
            participant_id=participant_id,
            passed=False,
            reference_mps=reference,
            error=f"Reference gait speed must be positive, got {reference}",
        )

    tolerance = (
        default_gait_speed_tolerance_relative()
        if tolerance_relative is None
        else tolerance_relative
    )
    try:
        extracted = median_stride_gait_speed_mps(record)
    except AssertionError as exc:
        return GaitSpeedReferenceAudit(
            participant_id=participant_id,
            passed=False,
            reference_mps=reference,
            tolerance_relative=tolerance,
            error=str(exc),
        )

    relative_error = abs(extracted - reference) / reference
    passed = relative_error <= tolerance
    return GaitSpeedReferenceAudit(
        participant_id=participant_id,
        passed=passed,
        extracted_mps=extracted,
        reference_mps=reference,
        relative_error=relative_error,
        tolerance_relative=tolerance,
    )


def assert_gait_speed_matches_reference(
    record: RecordFeatures,
    participant_id: str,
    *,
    reference_speed_mps: Optional[float] = None,
    tolerance_relative: Optional[float] = None,
) -> None:
    """Compare median extracted stride GAIT_SPEED to clinical lab-walk reference."""
    result = audit_gait_speed_matches_reference(
        record,
        participant_id,
        reference_speed_mps=reference_speed_mps,
        tolerance_relative=tolerance_relative,
    )
    if result.error and result.extracted_mps is None:
        raise AssertionError(result.error)
    if not result.passed:
        if result.error:
            raise AssertionError(result.error)
        raise AssertionError(
            f"GAIT_SPEED mismatch for {participant_id}: "
            f"median extracted={result.extracted_mps:.3f} m/s, "
            f"clinical reference={result.reference_mps:.3f} m/s, "
            f"relative error={result.relative_error:.1%} "
            f"(tolerance={result.tolerance_relative:.1%})"
        )
