import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.app.bootstrap import build_visualization_data_service
from src.data_visualization.suite.plots.feature_plot_engine import FeaturePlotEngine

matplotlib.use("Agg")


def _resolve_feature_types(
    requested: str,
    available_feature_types: Iterable[FeatureType],
) -> List[FeatureType]:
    available = list(available_feature_types)
    if not requested.strip():
        return available
    requested_values = {item.strip() for item in requested.split(",") if item.strip()}
    return [item for item in available if item.value in requested_values]


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
) -> None:
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

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, str]] = []
    with PdfPages(output_pdf) as pdf:
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

    if summary_csv is not None:
        _write_summary_csv(summary_csv, summary_rows)


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
        help="Optional path for summary statistics CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_feature_report(
        sqlite_db_path=args.sqlite_db_path,
        output_pdf=args.output_pdf,
        basis=SampleBasis(args.basis),
        feature_type_filter=args.feature_types,
        max_features=args.max_features,
        summary_csv=args.summary_csv,
    )


if __name__ == "__main__":
    main()
