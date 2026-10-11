"""Aggregate per-record validation results into per-user and per-feature views.

Two rollups, matching the two questions worth asking of a feature set:

* per-user   -- can I trust this participant's features at all?
* per-feature -- is this feature type trustworthy across the cohort?
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.feature_validation.feature_validator import RecordValidationResult


@dataclass(frozen=True)
class PerUserValidation:
    """Finite and meaningful counts rolled up for one participant."""

    participant_id: str
    class_label: str
    feature_records: int
    usable_samples: int
    finite_values: int
    missing_values: int
    implausible_values: int
    atypical_values: int
    feature_types_scored: int
    feature_types_all_missing: List[str] = field(default_factory=list)
    feature_types_with_implausible: List[str] = field(default_factory=list)

    @property
    def total_values(self) -> int:
        return self.finite_values + self.missing_values

    @property
    def finite_fraction(self) -> float:
        if self.total_values == 0:
            return 0.0
        return self.finite_values / self.total_values

    @property
    def meaningful_fraction(self) -> float:
        if self.finite_values == 0:
            return 0.0
        good = self.finite_values - self.implausible_values - self.atypical_values
        return good / self.finite_values

    @property
    def implausible_fraction(self) -> float:
        if self.finite_values == 0:
            return 0.0
        return self.implausible_values / self.finite_values


@dataclass(frozen=True)
class PerFeatureValidation:
    """Finite and meaningful counts rolled up for one feature type."""

    feature_type: FeatureType
    unit: str
    has_range: bool
    participants: int
    usable_samples: int
    finite_values: int
    missing_values: int
    implausible_values: int
    atypical_values: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    worst_implausible: Optional[float] = None
    participants_with_implausible: List[str] = field(default_factory=list)
    reference_key: str = ""
    implausible_provenance: str = ""
    atypical_provenance: str = ""

    @property
    def total_values(self) -> int:
        return self.finite_values + self.missing_values

    @property
    def finite_fraction(self) -> float:
        if self.total_values == 0:
            return 0.0
        return self.finite_values / self.total_values

    @property
    def missing_fraction(self) -> float:
        return 1.0 - self.finite_fraction

    @property
    def meaningful_fraction(self) -> float:
        if self.finite_values == 0:
            return 0.0
        good = self.finite_values - self.implausible_values - self.atypical_values
        return good / self.finite_values

    @property
    def implausible_fraction(self) -> float:
        if self.finite_values == 0:
            return 0.0
        return self.implausible_values / self.finite_values


@dataclass
class ValidationSummary:
    """Both rollups for one sample basis."""

    basis: SampleBasis
    per_user: List[PerUserValidation] = field(default_factory=list)
    per_feature: List[PerFeatureValidation] = field(default_factory=list)
    uncovered_feature_types: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.per_user and not self.per_feature

    def summary_text(self) -> str:
        if self.is_empty:
            return "Basis: {}\nNo validated records.".format(self.basis.value)

        total_finite = sum(f.finite_values for f in self.per_feature)
        total_missing = sum(f.missing_values for f in self.per_feature)
        total_implausible = sum(f.implausible_values for f in self.per_feature)
        total_atypical = sum(f.atypical_values for f in self.per_feature)
        denominator = max(total_finite + total_missing, 1)

        lines = [
            "Basis: {}".format(self.basis.value),
            "Participants: {}".format(len(self.per_user)),
            "Feature types: {}".format(len(self.per_feature)),
            "Values checked: {}".format(total_finite + total_missing),
            "Finite: {} ({:.1%})".format(total_finite, total_finite / denominator),
            "Missing: {} ({:.1%})".format(total_missing, total_missing / denominator),
            "Implausible: {} ({:.2%} of finite)".format(
                total_implausible, total_implausible / max(total_finite, 1)
            ),
            "Atypical: {} ({:.2%} of finite)".format(
                total_atypical, total_atypical / max(total_finite, 1)
            ),
        ]

        worst_missing = sorted(self.per_feature, key=lambda f: f.finite_fraction)[:3]
        if worst_missing and worst_missing[0].missing_fraction > 0:
            lines.append(
                "Highest missingness: "
                + ", ".join(
                    "{} ({:.1%})".format(f.feature_type.value, f.missing_fraction)
                    for f in worst_missing
                    if f.missing_fraction > 0
                )
            )

        offenders = sorted(
            (f for f in self.per_feature if f.implausible_values > 0),
            key=lambda f: f.implausible_fraction,
            reverse=True,
        )[:3]
        if offenders:
            lines.append(
                "Most implausible: "
                + ", ".join(
                    "{} ({:.2%}, worst={:g})".format(
                        f.feature_type.value,
                        f.implausible_fraction,
                        f.worst_implausible if f.worst_implausible is not None else float("nan"),
                    )
                    for f in offenders
                )
            )

        if self.uncovered_feature_types:
            lines.append(
                "No range defined for {} feature types: {}".format(
                    len(self.uncovered_feature_types),
                    ", ".join(sorted(self.uncovered_feature_types)[:5]),
                )
            )
        return "\n".join(lines)


def aggregate_validation(
    results: Sequence[RecordValidationResult], basis: SampleBasis
) -> ValidationSummary:
    """Roll per-record results up into per-user and per-feature tables."""
    if not results:
        return ValidationSummary(basis=basis)

    by_participant: Dict[str, List[RecordValidationResult]] = defaultdict(list)
    for result in results:
        by_participant[result.participant_id].append(result)

    per_user: List[PerUserValidation] = []
    for participant_id, participant_results in sorted(by_participant.items()):
        all_missing: List[str] = []
        with_implausible: List[str] = []
        finite = missing = implausible = atypical = usable = 0
        scored = 0
        for result in participant_results:
            usable += result.usable_samples
            for counts in result.counts:
                scored += 1
                finite += counts.finite_count
                missing += counts.missing_count
                implausible += counts.implausible_count
                atypical += counts.atypical_count
                name = str(getattr(counts.feature_type, "value", counts.feature_type))
                if counts.finite_count == 0 and counts.usable_samples > 0:
                    all_missing.append(name)
                if counts.implausible_count > 0:
                    with_implausible.append(name)

        per_user.append(
            PerUserValidation(
                participant_id=participant_id,
                class_label=participant_results[0].class_label,
                feature_records=len(participant_results),
                usable_samples=usable,
                finite_values=finite,
                missing_values=missing,
                implausible_values=implausible,
                atypical_values=atypical,
                feature_types_scored=scored,
                feature_types_all_missing=sorted(set(all_missing)),
                feature_types_with_implausible=sorted(set(with_implausible)),
            )
        )

    accumulator: Dict[object, Dict[str, object]] = {}
    for result in results:
        for counts in result.counts:
            bucket = accumulator.setdefault(
                counts.feature_type,
                {
                    "unit": counts.unit,
                    "has_range": counts.has_range,
                    "participants": set(),
                    "usable": 0,
                    "finite": 0,
                    "missing": 0,
                    "implausible": 0,
                    "atypical": 0,
                    "min": None,
                    "max": None,
                    "worst": None,
                    "offenders": set(),
                    "reference_key": counts.reference_key,
                    "implausible_provenance": counts.implausible_provenance,
                    "atypical_provenance": counts.atypical_provenance,
                },
            )
            bucket["participants"].add(result.participant_id)
            bucket["usable"] = int(bucket["usable"]) + counts.usable_samples
            bucket["finite"] = int(bucket["finite"]) + counts.finite_count
            bucket["missing"] = int(bucket["missing"]) + counts.missing_count
            bucket["implausible"] = int(bucket["implausible"]) + counts.implausible_count
            bucket["atypical"] = int(bucket["atypical"]) + counts.atypical_count
            if counts.min_value is not None:
                current = bucket["min"]
                bucket["min"] = (
                    counts.min_value if current is None else min(float(current), counts.min_value)
                )
            if counts.max_value is not None:
                current = bucket["max"]
                bucket["max"] = (
                    counts.max_value if current is None else max(float(current), counts.max_value)
                )
            if counts.worst_implausible is not None:
                current = bucket["worst"]
                if current is None or abs(counts.worst_implausible) > abs(float(current)):
                    bucket["worst"] = counts.worst_implausible
            if counts.implausible_count > 0:
                bucket["offenders"].add(result.participant_id)

    per_feature: List[PerFeatureValidation] = []
    for feature_type, bucket in accumulator.items():
        per_feature.append(
            PerFeatureValidation(
                feature_type=feature_type,
                unit=str(bucket["unit"]),
                has_range=bool(bucket["has_range"]),
                participants=len(bucket["participants"]),
                usable_samples=int(bucket["usable"]),
                finite_values=int(bucket["finite"]),
                missing_values=int(bucket["missing"]),
                implausible_values=int(bucket["implausible"]),
                atypical_values=int(bucket["atypical"]),
                min_value=bucket["min"],
                max_value=bucket["max"],
                worst_implausible=bucket["worst"],
                participants_with_implausible=sorted(bucket["offenders"]),
                reference_key=str(bucket["reference_key"]),
                implausible_provenance=str(bucket["implausible_provenance"]),
                atypical_provenance=str(bucket["atypical_provenance"]),
            )
        )
    per_feature.sort(key=lambda f: str(getattr(f.feature_type, "value", f.feature_type)))

    uncovered = sorted(
        {
            ftype.value
            for result in results
            for ftype in result.uncovered_feature_types
        }
    )

    return ValidationSummary(
        basis=basis,
        per_user=sorted(per_user, key=lambda u: u.participant_id),
        per_feature=per_feature,
        uncovered_feature_types=uncovered,
    )
