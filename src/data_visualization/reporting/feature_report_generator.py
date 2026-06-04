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


def _render_cover_page(
    pdf: PdfPages,
    basis: SampleBasis,
    generated_at: datetime,
    num_feature_records: int,
    num_feature_types: int,
) -> None:
    figure = Figure(figsize=(11, 8.5))
    axis = figure.add_subplot(111)
    axis.set_axis_off()
    lines = [
        "Feature Population Report",
        "",
        f"Sample basis: {basis.value}",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Feature records: {num_feature_records}",
        f"Feature types: {num_feature_types}",
    ]
    axis.text(
        0.5,
        0.6,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=16,
        linespacing=1.8,
    )
    figure.tight_layout()
    pdf.savefig(figure)


def _write_summary_csv(
    summary_path: Path,
    summary_rows: List[Dict[str, str]],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feature_type",
                "basis",
                "class_label",
                "count",
                "mean",
                "median",
                "std",
                "iqr",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def generate_feature_report(
    sqlite_db_path: Path,
    output_pdf: Path,
    basis: SampleBasis,
    feature_type_filter: str = "",
    max_features: int = 0,
    summary_csv: Optional[Path] = None,
) -> Tuple[Path, Path]:
    data_service = build_visualization_data_service(sqlite_db_path)
    feature_ids = data_service.list_feature_ids()
    if not feature_ids:
        raise ValueError("No feature records are indexed in the database.")

    available_feature_types = data_service.list_feature_types_for_basis(feature_ids[0], basis)
    target_feature_types = _resolve_feature_types(feature_type_filter, available_feature_types)
    if max_features > 0:
        target_feature_types = target_feature_types[:max_features]
    if not target_feature_types:
        raise ValueError("No matching feature types found for report generation.")

    generated_at = datetime.now()
    resolved_pdf = _timestamped_path(output_pdf, basis, generated_at, ".pdf")
    resolved_csv = (
        summary_csv
        if summary_csv is not None
        else _timestamped_path(output_pdf, basis, generated_at, ".csv")
    )
    resolved_pdf.parent.mkdir(parents=True, exist_ok=True)

    population = data_service.collect_population_feature_matrix(basis, target_feature_types)

    summary_rows: List[Dict[str, str]] = []
    with PdfPages(resolved_pdf) as pdf:
        _render_cover_page(
            pdf,
            basis=basis,
            generated_at=generated_at,
            num_feature_records=len(feature_ids),
            num_feature_types=len(target_feature_types),
        )

        if not population.is_empty:
            correlation_figure = Figure(figsize=(11, 8.5))
            FeaturePlotEngine(correlation_figure).render_feature_correlation_heatmap(
                feature_types=population.feature_types,
                feature_matrix=population.matrix,
                basis=basis,
            )
            pdf.savefig(correlation_figure)

            separation_figure = Figure(figsize=(11, 8.5))
            FeaturePlotEngine(separation_figure).render_class_separation(
                feature_matrix=population.matrix,
                row_class_labels=population.row_class_labels,
                feature_types=population.feature_types,
                basis=basis,
            )
            pdf.savefig(separation_figure)

            coverage_figure = Figure(figsize=(11, 8.5))
            FeaturePlotEngine(coverage_figure).render_feature_coverage(
                feature_matrix=population.matrix,
                row_class_labels=population.row_class_labels,
                feature_types=population.feature_types,
                basis=basis,
            )
            pdf.savefig(coverage_figure)

        for feature_type in target_feature_types:
            grouped_values = data_service.collect_feature_values_by_class(
                basis=basis,
                feature_type=feature_type,
            )
            figure = Figure(figsize=(11, 8.5))
            plot_engine = FeaturePlotEngine(figure)
            summary = plot_engine.render_class_violin(
                grouped_values=grouped_values,
                feature_type=feature_type,
                basis=basis,
            )
            pdf.savefig(figure)
            for class_label, stats in summary.items():
                summary_rows.append(
                    {
                        "feature_type": feature_type.value,
                        "basis": basis.value,
                        "class_label": class_label,
                        "count": str(int(stats["count"])),
                        "mean": f"{stats['mean']:.6f}",
                        "median": f"{stats['median']:.6f}",
                        "std": f"{stats['std']:.6f}",
                        "iqr": f"{stats['iqr']:.6f}",
                    }
                )

    _write_summary_csv(resolved_csv, summary_rows)
    return resolved_pdf, resolved_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-based feature report PDF")
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
            "Optional path for summary statistics CSV. If omitted, the CSV is "
            "written to the PDF's parent directory with a matching basis/timestamp name."
        ),
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
    )
    print(f"Report PDF: {resolved_pdf}")
    print(f"Summary CSV: {resolved_csv}")


if __name__ == "__main__":
    main()
