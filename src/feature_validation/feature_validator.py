"""Per-record validation of generated features: finite values and physiological range.

Operates on ``RecordFeatures`` already loaded from disk. Two independent questions
are answered per feature type:

* *finite* -- of the usable sample rows, how many carry a real number?
* *meaningful* -- of those real numbers, how many fall within physiological range?

Padded sample slots are excluded up front via the canonical usable-sample helpers,
so structural padding is never counted as missing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data_model.features.bout_sample_layout import (
    flatten_bout_features,
    usable_sample_mask,
)
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.feature_validation.range_catalog import (
    ATYPICAL,
    IMPLAUSIBLE,
    FeatureRange,
    RangeCatalog,
)


@dataclass(frozen=True)
class FeatureValidationCounts:
    """Finite and range counts for one feature type within one record."""

    feature_type: FeatureType
    unit: str
    usable_samples: int
    finite_count: int
    implausible_count: int
    atypical_count: int
    has_range: bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median_value: Optional[float] = None
    worst_implausible: Optional[float] = None
    reference_key: str = ""
    implausible_provenance: str = ""
    atypical_provenance: str = ""

    @property
    def missing_count(self) -> int:
        return self.usable_samples - self.finite_count

    @property
    def in_range_count(self) -> int:
        return self.finite_count - self.implausible_count - self.atypical_count

    @property
    def finite_fraction(self) -> float:
        if self.usable_samples == 0:
            return 0.0
        return self.finite_count / self.usable_samples

    @property
    def meaningful_fraction(self) -> float:
        """Fraction of finite values inside every defined bound.

        Returns 0.0 when there are no finite values, and 1.0 when the feature has
        no range definition, so that absent coverage never reads as a failure.
        """
        if self.finite_count == 0:
            return 0.0
        if not self.has_range:
            return 1.0
        return self.in_range_count / self.finite_count


@dataclass(frozen=True)
class RecordValidationResult:
    """Validation counts for every feature type of one record and basis."""

    feature_id: str
    participant_id: str
    class_label: str
    basis: SampleBasis
    usable_samples: int
    counts: List[FeatureValidationCounts] = field(default_factory=list)
    uncovered_feature_types: List[FeatureType] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.usable_samples == 0 or not self.counts

    @property
    def implausible_total(self) -> int:
        return sum(c.implausible_count for c in self.counts)

    @property
    def atypical_total(self) -> int:
        return sum(c.atypical_count for c in self.counts)

    @property
    def missing_total(self) -> int:
        return sum(c.missing_count for c in self.counts)

    @property
    def finite_total(self) -> int:
        return sum(c.finite_count for c in self.counts)

    @property
    def mean_finite_fraction(self) -> float:
        if not self.counts:
            return 0.0
        return float(np.mean([c.finite_fraction for c in self.counts]))

    @property
    def mean_meaningful_fraction(self) -> float:
        scored = [c for c in self.counts if c.finite_count > 0]
        if not scored:
            return 0.0
        return float(np.mean([c.meaningful_fraction for c in scored]))


@dataclass(frozen=True)
class _TierTallies:
    implausible: int = 0
    atypical: int = 0
    worst_implausible: Optional[float] = None


def _classify_values(
    values: np.ndarray, feature_range: Optional[FeatureRange]
) -> _TierTallies:
    """Count tier violations and capture the most extreme implausible value."""
    if feature_range is None:
        return _TierTallies()

    implausible = 0
    atypical = 0
    worst: Optional[float] = None
    worst_distance = -1.0

    for value in values:
        tier = feature_range.classify(float(value))
        if tier == IMPLAUSIBLE:
            implausible += 1
            distance = _bound_distance(float(value), feature_range)
            if distance > worst_distance:
                worst_distance = distance
                worst = float(value)
        elif tier == ATYPICAL:
            atypical += 1

    return _TierTallies(implausible=implausible, atypical=atypical, worst_implausible=worst)


def _bound_distance(value: float, feature_range: FeatureRange) -> float:
    """Distance outside the implausible bounds, used to pick the worst offender."""
    bounds = feature_range.implausible
    if bounds is None:
        return 0.0
    if bounds.minimum is not None and value < bounds.minimum:
        return bounds.minimum - value
    if bounds.maximum is not None and value > bounds.maximum:
        return value - bounds.maximum
    return 0.0


def validate_record(
    record: RecordFeatures,
    basis: SampleBasis,
    catalog: RangeCatalog,
    participant_id: Optional[str] = None,
    class_label: str = "unknown",
) -> RecordValidationResult:
    """Validate one basis of one record against the range catalog."""
    bout = record.get_features_by_basis(basis)
    feature_id = record.feature_metadata.feature_identifier.value
    resolved_participant = (
        participant_id
        if participant_id is not None
        else record.feature_metadata.user_identifier.value
    )

    features = np.asarray(bout.features, dtype=float)
    if features.size == 0:
        return RecordValidationResult(
            feature_id=feature_id,
            participant_id=resolved_participant,
            class_label=class_label,
            basis=basis,
            usable_samples=0,
        )

    per_sample = flatten_bout_features(features)
    usable = per_sample[usable_sample_mask(per_sample)]
    usable_samples = int(usable.shape[0])

    counts: List[FeatureValidationCounts] = []
    uncovered: List[FeatureType] = []

    for column, feature_type in enumerate(bout.feature_names):
        values = usable[:, column]
        finite = values[np.isfinite(values)]
        feature_range = catalog.get(feature_type) if isinstance(feature_type, FeatureType) else None
        if feature_range is None and isinstance(feature_type, FeatureType):
            uncovered.append(feature_type)

        tallies = _classify_values(finite, feature_range)
        counts.append(
            FeatureValidationCounts(
                feature_type=feature_type,
                unit=feature_range.unit if feature_range else "",
                usable_samples=usable_samples,
                finite_count=int(finite.size),
                implausible_count=tallies.implausible,
                atypical_count=tallies.atypical,
                has_range=feature_range is not None,
                min_value=float(np.min(finite)) if finite.size else None,
                max_value=float(np.max(finite)) if finite.size else None,
                median_value=float(np.median(finite)) if finite.size else None,
                worst_implausible=tallies.worst_implausible,
                reference_key=feature_range.reference_key if feature_range else "",
                implausible_provenance=(
                    feature_range.implausible_provenance if feature_range else ""
                ),
                atypical_provenance=feature_range.atypical_provenance if feature_range else "",
            )
        )

    return RecordValidationResult(
        feature_id=feature_id,
        participant_id=resolved_participant,
        class_label=class_label,
        basis=basis,
        usable_samples=usable_samples,
        counts=counts,
        uncovered_feature_types=uncovered,
    )


def validate_records(
    records: Sequence[RecordFeatures],
    basis: SampleBasis,
    catalog: RangeCatalog,
    class_labels: Optional[Dict[str, str]] = None,
) -> List[RecordValidationResult]:
    """Validate many records for one basis."""
    labels = class_labels or {}
    results = []
    for record in records:
        participant_id = record.feature_metadata.user_identifier.value
        results.append(
            validate_record(
                record=record,
                basis=basis,
                catalog=catalog,
                participant_id=participant_id,
                class_label=labels.get(participant_id, "unknown"),
            )
        )
    return results
