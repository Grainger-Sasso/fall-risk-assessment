"""Shared collection logic for post-hoc feature validation.

Both the CLI orchestrator and the visualization plugin drive validation through
this module so the numbers on screen and the numbers in the CSV cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from src.data_model.features.record_features import RecordFeatures
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.feature_validation.aggregation import ValidationSummary, aggregate_validation
from src.feature_validation.feature_validator import (
    RecordValidationResult,
    validate_record,
)
from src.feature_validation.index_integrity import (
    IndexIntegrityReport,
    audit_index_integrity,
)
from src.feature_validation.range_catalog import (
    RangeCatalog,
    default_range_catalog,
)

if TYPE_CHECKING:
    from src.data_visualization.suite.services.visualization_data_service import (
        VisualizationDataService,
    )

UNKNOWN_CLASS = "unknown"


@dataclass
class ValidationRun:
    """Everything one validation pass produces."""

    integrity: IndexIntegrityReport
    summaries: List[ValidationSummary] = field(default_factory=list)
    catalog_schema_version: str = ""
    catalog_feature_count: int = 0
    load_warnings: List[str] = field(default_factory=list)

    def summary_for(self, basis: SampleBasis) -> Optional[ValidationSummary]:
        for summary in self.summaries:
            if summary.basis == basis:
                return summary
        return None


def load_records_with_labels(
    data_service: VisualizationDataService,
) -> Tuple[List[Tuple[RecordFeatures, str]], List[str]]:
    """Load every readable feature record with its participant class label.

    Records that fail to load are skipped and reported as warnings; the index
    integrity audit is responsible for explaining why they failed.
    """
    loaded: List[Tuple[RecordFeatures, str]] = []
    warnings: List[str] = []

    for feature_id in data_service.list_feature_ids():
        try:
            record = data_service.load_features(feature_id)
        except Exception as exc:
            warnings.append(
                "{}: {}: {}".format(feature_id.value, type(exc).__name__, exc)
            )
            continue

        class_label = UNKNOWN_CLASS
        try:
            user = data_service.load_user_for_feature(feature_id)
            if user is not None:
                class_label = user.clinical_demographic_data.faller_status.value
        except Exception:
            pass
        loaded.append((record, class_label))

    return loaded, warnings


def build_summaries(
    records: List[Tuple[RecordFeatures, str]],
    bases: List[SampleBasis],
    catalog: RangeCatalog,
) -> Tuple[List[ValidationSummary], List[str]]:
    """Validate loaded records for each basis and roll the results up."""
    summaries: List[ValidationSummary] = []
    warnings: List[str] = []

    for basis in bases:
        results: List[RecordValidationResult] = []
        for record, class_label in records:
            try:
                results.append(
                    validate_record(
                        record=record,
                        basis=basis,
                        catalog=catalog,
                        class_label=class_label,
                    )
                )
            except Exception as exc:
                warnings.append(
                    "{} / {}: {}: {}".format(
                        record.feature_metadata.feature_identifier.value,
                        basis.value,
                        type(exc).__name__,
                        exc,
                    )
                )
        summaries.append(aggregate_validation(results, basis))

    return summaries, warnings


def run_feature_validation(
    data_service: VisualizationDataService,
    bases: Optional[List[SampleBasis]] = None,
    catalog: Optional[RangeCatalog] = None,
    feature_payload_roots: Optional[List[Path]] = None,
) -> ValidationRun:
    """Audit index integrity and validate every readable feature record."""
    resolved_catalog = catalog or default_range_catalog()
    resolved_bases = bases or list(SampleBasis)

    integrity = audit_index_integrity(
        db_manager=data_service.db_manager,
        feature_payload_roots=feature_payload_roots,
    )
    records, load_warnings = load_records_with_labels(data_service)
    summaries, validate_warnings = build_summaries(records, resolved_bases, resolved_catalog)

    return ValidationRun(
        integrity=integrity,
        summaries=summaries,
        catalog_schema_version=resolved_catalog.schema_version,
        catalog_feature_count=len(resolved_catalog.ranges),
        load_warnings=load_warnings + validate_warnings,
    )
