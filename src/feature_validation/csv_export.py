"""CSV export for validation results.

Follows the conventions already used by ``data_visualization.reporting``: stdlib
``csv``, timestamped filenames, parents created on demand, UTF-8 output.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List

from src.feature_validation.aggregation import ValidationSummary
from src.feature_validation.index_integrity import IndexIntegrityReport

PER_FEATURE_FIELDS = [
    "basis",
    "feature_type",
    "unit",
    "has_range",
    "implausible_provenance",
    "atypical_provenance",
    "reference",
    "participants",
    "usable_samples",
    "total_values",
    "finite_values",
    "missing_values",
    "finite_fraction",
    "missing_fraction",
    "implausible_values",
    "atypical_values",
    "implausible_fraction",
    "meaningful_fraction",
    "min_value",
    "max_value",
    "worst_implausible",
    "participants_with_implausible",
]

PER_USER_FIELDS = [
    "basis",
    "participant_id",
    "class_label",
    "feature_records",
    "usable_samples",
    "total_values",
    "finite_values",
    "missing_values",
    "finite_fraction",
    "implausible_values",
    "atypical_values",
    "implausible_fraction",
    "meaningful_fraction",
    "feature_types_scored",
    "feature_types_all_missing",
    "feature_types_with_implausible",
]

INTEGRITY_FIELDS = ["metric", "value", "detail"]


def timestamped_path(base: Path, label: str, generated_at: datetime) -> Path:
    """Return ``{stem}_{label}_{YYYYMMDD_HHMMSS}{suffix}`` beside ``base``."""
    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    suffix = base.suffix or ".csv"
    return base.with_name("{}_{}_{}{}".format(base.stem, label, stamp, suffix))


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.6g}".format(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def write_per_feature_csv(path: Path, summaries: List[ValidationSummary]) -> Path:
    """Write the per-feature-type rollup for every basis into one CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_FEATURE_FIELDS)
        writer.writeheader()
        for summary in summaries:
            for feature in summary.per_feature:
                writer.writerow(
                    {
                        "basis": summary.basis.value,
                        "feature_type": _fmt(
                            getattr(feature.feature_type, "value", feature.feature_type)
                        ),
                        "unit": feature.unit,
                        "has_range": _fmt(feature.has_range),
                        "implausible_provenance": feature.implausible_provenance,
                        "atypical_provenance": feature.atypical_provenance,
                        "reference": feature.reference_key,
                        "participants": feature.participants,
                        "usable_samples": feature.usable_samples,
                        "total_values": feature.total_values,
                        "finite_values": feature.finite_values,
                        "missing_values": feature.missing_values,
                        "finite_fraction": _fmt(feature.finite_fraction),
                        "missing_fraction": _fmt(feature.missing_fraction),
                        "implausible_values": feature.implausible_values,
                        "atypical_values": feature.atypical_values,
                        "implausible_fraction": _fmt(feature.implausible_fraction),
                        "meaningful_fraction": _fmt(feature.meaningful_fraction),
                        "min_value": _fmt(feature.min_value),
                        "max_value": _fmt(feature.max_value),
                        "worst_implausible": _fmt(feature.worst_implausible),
                        "participants_with_implausible": _fmt(
                            feature.participants_with_implausible
                        ),
                    }
                )
    return path


def write_per_user_csv(path: Path, summaries: List[ValidationSummary]) -> Path:
    """Write the per-participant rollup for every basis into one CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_USER_FIELDS)
        writer.writeheader()
        for summary in summaries:
            for user in summary.per_user:
                writer.writerow(
                    {
                        "basis": summary.basis.value,
                        "participant_id": user.participant_id,
                        "class_label": user.class_label,
                        "feature_records": user.feature_records,
                        "usable_samples": user.usable_samples,
                        "total_values": user.total_values,
                        "finite_values": user.finite_values,
                        "missing_values": user.missing_values,
                        "finite_fraction": _fmt(user.finite_fraction),
                        "implausible_values": user.implausible_values,
                        "atypical_values": user.atypical_values,
                        "implausible_fraction": _fmt(user.implausible_fraction),
                        "meaningful_fraction": _fmt(user.meaningful_fraction),
                        "feature_types_scored": user.feature_types_scored,
                        "feature_types_all_missing": _fmt(user.feature_types_all_missing),
                        "feature_types_with_implausible": _fmt(
                            user.feature_types_with_implausible
                        ),
                    }
                )
    return path


def write_integrity_csv(path: Path, report: IndexIntegrityReport) -> Path:
    """Write index/payload drift as one metric per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("indexed_feature_records", report.indexed_feature_count, ""),
        ("loadable_feature_records", report.loadable_feature_count, ""),
        ("index_overstatement", report.index_overstatement, ""),
        (
            "dangling_index_rows",
            report.dangling_count,
            ";".join(report.dangling_feature_ids),
        ),
        (
            "unreadable_payloads",
            report.unreadable_count,
            ";".join(
                "{} ({})".format(fid, err)
                for fid, err in sorted(report.unreadable_feature_ids.items())
            ),
        ),
        (
            "orphaned_payload_dirs",
            report.orphaned_count,
            ";".join(report.orphaned_payload_dirs),
        ),
        ("indexed_imu_records", report.indexed_imu_count, ""),
        (
            "imu_records_without_features",
            len(report.imu_ids_without_features),
            ";".join(report.imu_ids_without_features),
        ),
        (
            "participants_with_features",
            len(report.participants_with_features),
            "",
        ),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(INTEGRITY_FIELDS)
        for metric, value, detail in rows:
            writer.writerow([metric, value, detail])
    return path
