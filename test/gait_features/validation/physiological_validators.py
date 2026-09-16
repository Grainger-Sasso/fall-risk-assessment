from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.bout_sample_layout import flatten_bout_features, usable_sample_mask
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from test.gait_features.validation.physiological_ranges import (
    EPOCH_PHYSIOLOGICAL_RANGES,
    EPOCH_POPULATION_ANCHORS,
    STRIDE_PHYSIOLOGICAL_RANGES,
    STRIDE_POPULATION_ANCHORS,
    PhysiologicalRange,
)


@dataclass(frozen=True)
class RangeViolation:
    feature_type: FeatureType
    description: str
    out_of_range_count: int
    finite_count: int
    example_values: Tuple[float, ...]
    bounds: str


@dataclass(frozen=True)
class PhysiologicalAuditResult:
    """Non-throwing physiological plausibility outcome for reporting."""

    basis: str
    passed: bool
    violations: Tuple[RangeViolation, ...] = ()
    degenerate_row_indices: Tuple[int, ...] = ()
    error: Optional[str] = None

    @property
    def violation_count(self) -> int:
        return len(self.violations)


def _format_bounds(spec: PhysiologicalRange) -> str:
    parts: List[str] = []
    if spec.strictly_positive:
        parts.append("> 0")
    if spec.minimum is not None:
        parts.append(f"min={spec.minimum}")
    if spec.maximum is not None:
        parts.append(f"max={spec.maximum}")
    return ", ".join(parts)


def _validate_bout_features_against_ranges(
    bout: BoutFeatures,
    range_catalog: Dict[FeatureType, PhysiologicalRange],
    *,
    min_finite_fraction: float = 0.5,
) -> List[RangeViolation]:
    per_sample = flatten_bout_features(bout.features)
    mask = usable_sample_mask(per_sample)
    usable = per_sample[mask]
    if usable.size == 0:
        raise AssertionError(f"No usable {bout.sample_basis.value} samples to validate")

    violations: List[RangeViolation] = []
    for feature_index, feature_type in enumerate(bout.feature_names):
        if feature_type not in range_catalog:
            continue
        spec = range_catalog[feature_type]
        column = usable[:, feature_index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue

        in_range = np.array([spec.contains(float(value)) for value in finite], dtype=bool)
        out_count = int(finite.size - np.sum(in_range))
        if out_count == 0:
            continue

        finite_fraction = float(finite.size / max(1, usable.shape[0]))
        if finite_fraction < min_finite_fraction:
            # Sparse finite values may reflect partial SKDH output; still report if any violate.
            pass

        examples = tuple(float(value) for value in finite[~in_range][:3])
        violations.append(
            RangeViolation(
                feature_type=feature_type,
                description=spec.description,
                out_of_range_count=out_count,
                finite_count=int(finite.size),
                example_values=examples,
                bounds=_format_bounds(spec),
            )
        )

    return violations


def _find_degenerate_rows(
    bout: BoutFeatures,
    anchor_features: Sequence[FeatureType],
) -> List[int]:
    per_sample = flatten_bout_features(bout.features)
    mask = usable_sample_mask(per_sample)
    usable = per_sample[mask]
    if usable.size == 0:
        return []

    name_to_index = {feature: idx for idx, feature in enumerate(bout.feature_names)}
    anchor_indices = [name_to_index[item] for item in anchor_features if item in name_to_index]
    if not anchor_indices:
        return []

    degenerate_rows: List[int] = []
    for row_index, row in enumerate(usable):
        finite_values = row[np.isfinite(row)]
        if finite_values.size == 0:
            continue
        if np.all(finite_values == 0.0):
            degenerate_rows.append(row_index)
            continue
        anchors = row[anchor_indices]
        anchor_finite = anchors[np.isfinite(anchors)]
        if anchor_finite.size > 0 and np.all(anchor_finite == 0.0):
            degenerate_rows.append(row_index)
    return degenerate_rows


def _assert_no_degenerate_all_zero_rows(
    bout: BoutFeatures,
    anchor_features: Sequence[FeatureType],
) -> None:
    degenerate_rows = _find_degenerate_rows(bout, anchor_features)
    if degenerate_rows:
        raise AssertionError(
            f"{bout.sample_basis.value} samples contain degenerate all-zero anchor features "
            f"at usable row indices {degenerate_rows[:5]}"
            f"{'...' if len(degenerate_rows) > 5 else ''} "
            f"(anchors: {[item.value for item in anchor_features]})"
        )


def _raise_if_violations(
    violations: Iterable[RangeViolation],
    *,
    context: str,
    max_reported: int = 5,
) -> None:
    violation_list = list(violations)
    if not violation_list:
        return

    lines = [f"{context} physiological plausibility failures:"]
    for item in violation_list[:max_reported]:
        examples = ", ".join(f"{value:.4g}" for value in item.example_values)
        lines.append(
            f"  - {item.feature_type.value}: {item.out_of_range_count}/"
            f"{item.finite_count} finite values out of range "
            f"({item.bounds}); examples: {examples}"
        )
    if len(violation_list) > max_reported:
        lines.append(f"  ... and {len(violation_list) - max_reported} more feature(s)")
    raise AssertionError("\n".join(lines))


def _audit_bout_physiological_plausibility(
    bout: BoutFeatures,
    *,
    basis_label: str,
    range_catalog: Dict[FeatureType, PhysiologicalRange],
    anchor_features: Sequence[FeatureType],
) -> PhysiologicalAuditResult:
    try:
        degenerate_rows = _find_degenerate_rows(bout, anchor_features)
        violations = _validate_bout_features_against_ranges(bout, range_catalog)
        passed = not degenerate_rows and not violations
        return PhysiologicalAuditResult(
            basis=basis_label,
            passed=passed,
            violations=tuple(violations),
            degenerate_row_indices=tuple(degenerate_rows),
        )
    except AssertionError as exc:
        return PhysiologicalAuditResult(
            basis=basis_label,
            passed=False,
            error=str(exc),
        )


def audit_stride_physiological_plausibility(record: RecordFeatures) -> PhysiologicalAuditResult:
    """Return stride physiological plausibility outcome without raising."""
    return _audit_bout_physiological_plausibility(
        record.stride_features,
        basis_label="stride",
        range_catalog=STRIDE_PHYSIOLOGICAL_RANGES,
        anchor_features=STRIDE_POPULATION_ANCHORS,
    )


def audit_epoch_physiological_plausibility(record: RecordFeatures) -> PhysiologicalAuditResult:
    """Return epoch physiological plausibility outcome without raising."""
    return _audit_bout_physiological_plausibility(
        record.epoch_features,
        basis_label="epoch",
        range_catalog=EPOCH_PHYSIOLOGICAL_RANGES,
        anchor_features=EPOCH_POPULATION_ANCHORS,
    )


def assert_stride_physiological_plausibility(record: RecordFeatures) -> None:
    """Validate finite stride feature values fall within walking-plausible bounds."""
    result = audit_stride_physiological_plausibility(record)
    if result.error:
        raise AssertionError(result.error)
    if result.degenerate_row_indices:
        _assert_no_degenerate_all_zero_rows(record.stride_features, STRIDE_POPULATION_ANCHORS)
    _raise_if_violations(result.violations, context="Stride")


def assert_epoch_physiological_plausibility(record: RecordFeatures) -> None:
    """Validate finite epoch feature values fall within walking-plausible bounds."""
    result = audit_epoch_physiological_plausibility(record)
    if result.error:
        raise AssertionError(result.error)
    if result.degenerate_row_indices:
        _assert_no_degenerate_all_zero_rows(record.epoch_features, EPOCH_POPULATION_ANCHORS)
    _raise_if_violations(result.violations, context="Epoch")
