"""Structured feature-quality reports at feature and participant levels.

Usable-sample semantics match classification and population analytics:
a sample row is *usable* when at least one feature value is finite (structural
NaN padding in the ``(bout, feature, sample)`` tensor is excluded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.data_visualization.suite.services.visualization_data_service import (
    PerRecordCoverageMatrix,
    VisualizationDataService,
)


@dataclass
class FeatureLevelMetric:
    """Population-level quality statistics for one feature type."""

    feature_type: FeatureType
    valid_count: int
    total_usable_samples: int
    valid_fraction: float
    missing_fraction: float
    cohens_d: Optional[float] = None
    per_class_valid: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class FeatureLevelQualityReport:
    """Feature-centric quality report for a single sampling basis."""

    basis: SampleBasis
    feature_types: List[FeatureType]
    metrics: List[FeatureLevelMetric]
    correlation_matrix: np.ndarray
    separation: Dict[FeatureType, float]
    classes: List[str]
    total_usable_samples: int
    num_participants: int
    num_feature_records: int

    @property
    def is_empty(self) -> bool:
        return not self.feature_types or self.total_usable_samples == 0


@dataclass
class ParticipantLevelMetric:
    """Participant-centric quality statistics for one feature record."""

    participant_id: str
    class_label: str
    feature_id: str
    usable_sample_count: int
    tensor_slot_count: int
    padded_slot_count: int
    mean_coverage: float
    per_feature_fractions: np.ndarray
    count_consistent: bool


@dataclass
class ParticipantLevelQualityReport:
    """Participant-centric quality report for a single sampling basis."""

    basis: SampleBasis
    feature_types: List[FeatureType]
    participants: List[ParticipantLevelMetric]
    counts_by_class: Dict[str, np.ndarray]
    inconsistent_count_records: List[str]
    num_feature_records: int

    @property
    def is_empty(self) -> bool:
        return not self.participants


def build_participant_metrics(
    coverage: PerRecordCoverageMatrix,
    basis: SampleBasis,
    data_service: VisualizationDataService,
) -> List[ParticipantLevelMetric]:
    """Build per-participant metrics and verify usable-sample count consistency."""
    if coverage.is_empty:
        return []

    participants: List[ParticipantLevelMetric] = []
    for index, feature_id_str in enumerate(coverage.feature_ids):
        feature_id = FeatureIdentifier(feature_id_str)
        record = data_service.load_features(feature_id)
        bout = record.get_features_by_basis(basis)

        tensor_slots = int(bout.per_sample_matrix().shape[0])
        computed_count = int(bout.usable_sample_count)
        reported_count = int(coverage.usable_sample_counts[index])
        row_fractions = coverage.matrix[index, :]
        mean_coverage = float(np.nanmean(row_fractions)) if row_fractions.size else 0.0

        participants.append(
            ParticipantLevelMetric(
                participant_id=coverage.participant_ids[index],
                class_label=coverage.class_labels[index],
                feature_id=feature_id_str,
                usable_sample_count=computed_count,
                tensor_slot_count=tensor_slots,
                padded_slot_count=max(tensor_slots - computed_count, 0),
                mean_coverage=mean_coverage,
                per_feature_fractions=np.asarray(row_fractions, dtype=float),
                count_consistent=computed_count == reported_count,
            )
        )
    return participants


class FeatureQualityReportBuilder:
    """Build feature- and participant-level quality reports from indexed features."""

    def __init__(self, data_service: VisualizationDataService) -> None:
        self._data_service = data_service

    def build_feature_level(self, basis: SampleBasis) -> FeatureLevelQualityReport:
        from src.data_visualization.suite.plots.feature_plot_engine import FeaturePlotEngine

        population = self._data_service.collect_population_feature_matrix(basis)
        feature_ids = self._data_service.list_feature_ids()
        participant_ids = {
            self._data_service.load_features(feature_id).feature_metadata.user_identifier.value
            for feature_id in feature_ids
        }

        if population.is_empty:
            return FeatureLevelQualityReport(
                basis=basis,
                feature_types=population.feature_types,
                metrics=[],
                correlation_matrix=np.empty((0, 0)),
                separation={},
                classes=[],
                total_usable_samples=0,
                num_participants=len(participant_ids),
                num_feature_records=len(feature_ids),
            )

        coverage = FeaturePlotEngine.compute_feature_coverage(
            population.matrix,
            population.row_class_labels,
            population.feature_types,
        )
        separation, classes = FeaturePlotEngine.compute_class_separation(
            population.matrix,
            population.row_class_labels,
            population.feature_types,
        )
        correlation = FeaturePlotEngine.compute_correlation_matrix(population.matrix)

        metrics: List[FeatureLevelMetric] = []
        for feature_type in population.feature_types:
            stats = coverage.get(feature_type, {})
            valid = int(stats.get("valid", 0))
            total = int(stats.get("total", population.matrix.shape[0]))
            metrics.append(
                FeatureLevelMetric(
                    feature_type=feature_type,
                    valid_count=valid,
                    total_usable_samples=total,
                    valid_fraction=float(stats.get("valid_fraction", 0.0)),
                    missing_fraction=1.0 - float(stats.get("valid_fraction", 0.0)),
                    cohens_d=separation.get(feature_type),
                    per_class_valid=dict(stats.get("per_class", {})),
                )
            )

        return FeatureLevelQualityReport(
            basis=basis,
            feature_types=list(population.feature_types),
            metrics=metrics,
            correlation_matrix=correlation,
            separation=dict(separation),
            classes=list(classes),
            total_usable_samples=int(population.matrix.shape[0]),
            num_participants=len(participant_ids),
            num_feature_records=len(feature_ids),
        )

    def build_participant_level(self, basis: SampleBasis) -> ParticipantLevelQualityReport:
        coverage = self._data_service.collect_per_record_coverage(basis)
        counts_by_class = self._data_service.collect_sample_counts_by_basis_class().get(
            basis.value, {}
        )
        participants = build_participant_metrics(coverage, basis, self._data_service)
        inconsistent = [metric.feature_id for metric in participants if not metric.count_consistent]

        return ParticipantLevelQualityReport(
            basis=basis,
            feature_types=list(coverage.feature_types),
            participants=participants,
            counts_by_class=counts_by_class,
            inconsistent_count_records=inconsistent,
            num_feature_records=coverage.n_records,
        )


def feature_level_summary_text(report: FeatureLevelQualityReport) -> str:
    """Format a concise text summary for the feature-level report."""
    if report.is_empty:
        return "No usable samples found for this basis."

    lines = [
        f"Basis: {report.basis.value}",
        f"Feature records: {report.num_feature_records}",
        f"Participants: {report.num_participants}",
        f"Usable samples (padding excluded): {report.total_usable_samples}",
        f"Feature types: {len(report.feature_types)}",
    ]
    if report.classes:
        lines.append(f"Classes: {', '.join(report.classes)}")

    fully_missing = [
        metric.feature_type.value
        for metric in report.metrics
        if metric.valid_fraction == 0.0
    ]
    if fully_missing:
        lines.append(
            f"Fully missing features ({len(fully_missing)}): "
            f"{', '.join(fully_missing[:5])}"
        )
        if len(fully_missing) > 5:
            lines.append(f"  ... and {len(fully_missing) - 5} more")

    ranked_sep = sorted(
        report.separation.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    if ranked_sep:
        top = ranked_sep[0]
        lines.append(f"Top separability: {top[0].value} (Cohen's d = {top[1]:.3f})")

    high_missing = sorted(report.metrics, key=lambda m: m.missing_fraction, reverse=True)
    if high_missing and high_missing[0].missing_fraction > 0:
        worst = high_missing[0]
        lines.append(
            f"Highest missingness: {worst.feature_type.value} "
            f"({worst.missing_fraction:.1%} missing among usable samples)"
        )
    return "\n".join(lines)


def participant_level_summary_text(report: ParticipantLevelQualityReport) -> str:
    """Format a concise text summary for the participant-level report."""
    if report.is_empty:
        return "No feature records found for this basis."

    lines = [
        f"Basis: {report.basis.value}",
        f"Feature records: {report.num_feature_records}",
        f"Participants: {len(report.participants)}",
    ]
    if report.inconsistent_count_records:
        lines.append(
            "WARNING: usable_sample_count mismatch in "
            f"{len(report.inconsistent_count_records)} record(s)."
        )
    else:
        lines.append("Usable sample counts: stored tensor and coverage scan agree.")

    total_usable = sum(metric.usable_sample_count for metric in report.participants)
    total_padded = sum(metric.padded_slot_count for metric in report.participants)
    lines.append(f"Total usable samples: {total_usable}")
    if total_padded:
        lines.append(f"Total padded tensor slots (excluded): {total_padded}")

    for class_label, counts in sorted(report.counts_by_class.items()):
        if counts.size:
            lines.append(
                f"{class_label}: mean={float(np.mean(counts)):.1f}, "
                f"std={float(np.std(counts)):.1f} samples/participant (n={counts.size})"
            )

    worst = sorted(report.participants, key=lambda m: m.mean_coverage)
    if worst and worst[0].mean_coverage < 1.0:
        low = worst[0]
        lines.append(
            f"Lowest coverage: {low.participant_id} "
            f"(mean={low.mean_coverage:.2f}, usable={low.usable_sample_count})"
        )
    return "\n".join(lines)
