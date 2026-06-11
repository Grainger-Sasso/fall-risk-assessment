import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.app.bootstrap import build_visualization_data_service
from src.data_visualization.suite.plots.feature_plot_engine import FeaturePlotEngine
from src.data_visualization.suite.services.feature_quality_report import (
    FeatureQualityReportBuilder,
    feature_level_summary_text,
    participant_level_summary_text,
)

matplotlib.use("Agg")

FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def _resolve_feature_types(
    requested: str,
    available_feature_types: Iterable[FeatureType],
) -> List[FeatureType]:
    available = list(available_feature_types)
    if not requested.strip():
        return available
    requested_values = {item.strip() for item in requested.split(",") if item.strip()}
    return [item for item in available if item.value in requested_values]


def _timestamped_path(
    base_path: Path, basis: SampleBasis, generated_at: datetime, suffix: str
) -> Path:
    """Insert sample basis and generation timestamp into a file stem."""
    timestamp = generated_at.strftime(FILENAME_TIMESTAMP_FORMAT)
    return base_path.with_name(f"{base_path.stem}_{basis.value}_{timestamp}{suffix}")


def _render_text_page(pdf: PdfPages, title: str, body: str) -> None:
    figure = Figure(figsize=(11, 8.5))
    axis = figure.add_subplot(111)
    axis.set_axis_off()
    axis.text(
        0.05,
        0.95,
        f"{title}\n\n{body}",
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.5,
        family="monospace",
    )
    figure.tight_layout()
    pdf.savefig(figure)


def _write_feature_level_csv(
    summary_path: Path,
    rows: List[Dict[str, str]],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "report_level",
                "feature_type",
                "basis",
                "valid_count",
                "total_usable_samples",
                "valid_fraction",
                "missing_fraction",
                "cohens_d",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_participant_level_csv(
    summary_path: Path,
    rows: List[Dict[str, str]],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "report_level",
                "participant_id",
                "class_label",
                "feature_id",
                "basis",
                "usable_sample_count",
                "tensor_slot_count",
                "padded_slot_count",
                "mean_coverage",
                "count_consistent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_feature_report(
    sqlite_db_path: Path,
    output_pdf: Path,
    basis: SampleBasis,
    feature_type_filter: str = "",
    max_features: int = 0,
    summary_csv: Optional[Path] = None,
    include_participant_level: bool = True,
) -> Tuple[Path, Path]:
    data_service = build_visualization_data_service(sqlite_db_path)
    feature_ids = data_service.list_feature_ids()
    if not feature_ids:
        raise ValueError("No feature records are indexed in the database.")

    available_feature_types = data_service.list_feature_types_for_basis(feature_ids[0], basis)
    target_feature_types = _resolve_feature_types(feature_type_filter, available_feature_types)
    if max_features > 0:
        target_feature_types = target_feature_types[:max_features]

    generated_at = datetime.now()
    resolved_pdf = _timestamped_path(output_pdf, basis, generated_at, ".pdf")
    resolved_csv = (
        summary_csv
        if summary_csv is not None
        else _timestamped_path(output_pdf, basis, generated_at, ".csv")
    )
    resolved_pdf.parent.mkdir(parents=True, exist_ok=True)

    builder = FeatureQualityReportBuilder(data_service)
    feature_report = builder.build_feature_level(basis)
    if target_feature_types:
        allowed = {item for item in target_feature_types}
        feature_report.metrics = [
            metric for metric in feature_report.metrics if metric.feature_type in allowed
        ]
        feature_report.feature_types = [
            item for item in feature_report.feature_types if item in allowed
        ]

    participant_report = (
        builder.build_participant_level(basis) if include_participant_level else None
    )

    feature_rows: List[Dict[str, str]] = []
    participant_rows: List[Dict[str, str]] = []

    with PdfPages(resolved_pdf) as pdf:
        _render_text_page(
            pdf,
            title="Feature Quality Report",
            body=(
                f"Sample basis: {basis.value}\n"
                f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Feature records: {len(feature_ids)}\n"
                f"Includes participant-level section: {include_participant_level}"
            ),
        )

        _render_text_page(
            pdf,
            title="Feature level summary",
            body=feature_level_summary_text(feature_report),
        )

        if not feature_report.is_empty:
            for render_call in (
                lambda fig: FeaturePlotEngine(fig).render_feature_missingness(feature_report),
                lambda fig: FeaturePlotEngine(fig).render_feature_correlation_from_report(
                    feature_report
                ),
                lambda fig: FeaturePlotEngine(fig).render_feature_separability(feature_report),
            ):
                figure = Figure(figsize=(11, 8.5))
                render_call(figure)
                pdf.savefig(figure)

        for metric in feature_report.metrics:
            feature_rows.append(
                {
                    "report_level": "feature",
                    "feature_type": metric.feature_type.value,
                    "basis": basis.value,
                    "valid_count": str(metric.valid_count),
                    "total_usable_samples": str(metric.total_usable_samples),
                    "valid_fraction": f"{metric.valid_fraction:.6f}",
                    "missing_fraction": f"{metric.missing_fraction:.6f}",
                    "cohens_d": (
                        f"{metric.cohens_d:.6f}" if metric.cohens_d is not None else ""
                    ),
                }
            )

        if participant_report is not None:
            _render_text_page(
                pdf,
                title="Participant level summary",
                body=participant_level_summary_text(participant_report),
            )
            if not participant_report.is_empty:
                for render_call in (
                    lambda fig: FeaturePlotEngine(fig).render_participant_sample_counts(
                        participant_report
                    ),
                    lambda fig: FeaturePlotEngine(fig).render_participant_coverage_heatmap(
                        participant_report
                    ),
                ):
                    figure = Figure(figsize=(11, 8.5))
                    render_call(figure)
                    pdf.savefig(figure)

            for metric in participant_report.participants:
                participant_rows.append(
                    {
                        "report_level": "participant",
                        "participant_id": metric.participant_id,
                        "class_label": metric.class_label,
                        "feature_id": metric.feature_id,
                        "basis": basis.value,
                        "usable_sample_count": str(metric.usable_sample_count),
                        "tensor_slot_count": str(metric.tensor_slot_count),
                        "padded_slot_count": str(metric.padded_slot_count),
                        "mean_coverage": f"{metric.mean_coverage:.6f}",
                        "count_consistent": str(metric.count_consistent),
                    }
                )

    _write_feature_level_csv(resolved_csv, feature_rows)
    if participant_rows:
        participant_csv = resolved_csv.with_name(
            f"{resolved_csv.stem}_participants{resolved_csv.suffix}"
        )
        _write_participant_level_csv(participant_csv, participant_rows)

    return resolved_pdf, resolved_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate feature- and participant-level quality report PDF"
    )
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--basis", choices=["epoch", "stride"], default="epoch")
    parser.add_argument(
        "--feature-types",
        type=str,
        default="",
        help="Comma-separated FeatureType values. Empty means all available.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Limit number of feature types (0 means no limit).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional path for feature-level summary CSV. Participant rows are "
            "written alongside with a _participants suffix when enabled."
        ),
    )
    parser.add_argument(
        "--skip-participant-level",
        action="store_true",
        help="Omit participant-level pages and CSV from the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_pdf, resolved_csv = generate_feature_report(
        sqlite_db_path=args.sqlite_db_path,
        output_pdf=args.output_pdf,
        basis=SampleBasis(args.basis),
        feature_type_filter=args.feature_types,
        max_features=args.max_features,
        summary_csv=args.summary_csv,
        include_participant_level=not args.skip_participant_level,
    )
    print(f"Report PDF: {resolved_pdf}")
    print(f"Feature summary CSV: {resolved_csv}")


if __name__ == "__main__":
    main()
