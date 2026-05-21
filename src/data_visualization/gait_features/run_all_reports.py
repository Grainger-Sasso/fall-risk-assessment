"""
Run all gait-feature visualization reports with a single command.

This orchestrator runs:
1) dataset_report
2) aggregate_feature_report (+ AUC CSV)
3) feature_correlation_heatmap (+ correlation-groups text)
4) feature_histogram
5) executive_summary_report (PDF)

Usage:
    python -m src.data_visualization.gait_features.run_all_reports \
        --base-path /path/to/converted_data/ltmm_YYYY_MM_DD \
        [--results-dir /path/to/results] \
        [--hist-raw-feature-id raw_<uuid>] \
        [--hist-feature-type STRIDE_TIME]
"""

import argparse
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

from src.data_io.import_export.importers.features.raw.raw_feature_importer import (
    RawFeatureImporter,
)
from src.data_io.import_export.importers.registry.registry_importer import RegistryImporter
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier

from src.data_visualization.gait_features.aggregate_feature_report import (
    generate_report as generate_aggregate_feature_report,
)
from src.data_visualization.gait_features.dataset_report import (
    generate_report as generate_dataset_report,
)
from src.data_visualization.gait_features.feature_correlation_heatmap import (
    generate_feature_correlation_heatmap,
)
from src.data_visualization.gait_features.executive_summary_report import (
    generate_executive_summary,
)
from src.data_visualization.gait_features.feature_histogram import main as generate_histogram


def _dated_output_path(output_path: Path) -> Path:
    date_str = date.today().isoformat()
    return output_path.parent / f"{output_path.stem}_{date_str}{output_path.suffix}"


def _auto_select_histogram_target(base_path: Path) -> Tuple[str, str]:
    """
    Auto-select a raw feature ID and feature type for histogram generation.

    Selection strategy:
    - first raw feature ID in sorted registry order
    - first feature type with a non-NaN value in that record
    """
    raw_registry_path = base_path / "registries" / "raw_feature"
    registry_importer = RegistryImporter()
    raw_registry = registry_importer.import_data(raw_registry_path, RawFeatureIdentifier)
    raw_ids = sorted(raw_registry.registry.keys())
    if not raw_ids:
        raise ValueError("Raw feature registry is empty; cannot auto-select histogram input.")

    raw_feature_id = raw_ids[0]
    raw_importer = RawFeatureImporter()
    raw_entry = raw_importer.import_data(raw_registry.get_path_from_id(RawFeatureIdentifier(raw_feature_id)))

    for epoch in raw_entry.raw_epoch_features:
        for raw_feature in epoch.raw_features:
            if raw_feature.value is not None:
                try:
                    if raw_feature.value == raw_feature.value:
                        return raw_feature_id, raw_feature.feature_type.name
                except Exception:
                    continue

    # Fallback to a stable feature enum if no non-NaN values are found.
    return raw_feature_id, RawFeatureType.STRIDE_TIME.name


def run_all_reports(
    base_path: Path,
    results_dir: Optional[Path] = None,
    hist_raw_feature_id: Optional[str] = None,
    hist_feature_type: Optional[str] = None,
    corr_min_periods: int = 25,
    corr_group_threshold: float = 0.85,
    corr_annotate: bool = False,
) -> None:
    if results_dir is None:
        results_dir = base_path / "gait_feature_reports"
    results_dir.mkdir(parents=True, exist_ok=True)

    if hist_raw_feature_id is None or hist_feature_type is None:
        auto_raw_id, auto_feature_type = _auto_select_histogram_target(base_path)
        hist_raw_feature_id = hist_raw_feature_id or auto_raw_id
        hist_feature_type = hist_feature_type or auto_feature_type
        print(
            "Auto-selected histogram target: "
            f"raw_feature_id={hist_raw_feature_id}, feature_type={hist_feature_type}"
        )

    print("Running dataset report...")
    dataset_report_path = results_dir / "dataset_report.pdf"
    generate_dataset_report(base_path=base_path, output_path=dataset_report_path)

    print("Running aggregate feature report...")
    aggregate_report_path = results_dir / "aggregate_feature_report.pdf"
    auc_summary_path = results_dir / "aggregate_feature_auc_summary.csv"
    generate_aggregate_feature_report(
        base_path=base_path,
        output_path=aggregate_report_path,
        auc_output_path=auc_summary_path,
    )

    print("Running feature correlation heatmap...")
    correlation_heatmap_path = results_dir / "feature_correlation_heatmap.png"
    correlation_groups_path = results_dir / "feature_correlation_groups.txt"
    generate_feature_correlation_heatmap(
        base_path=base_path,
        output_path=correlation_heatmap_path,
        min_periods=corr_min_periods,
        annotate=corr_annotate,
        group_threshold=corr_group_threshold,
        groups_output_path=correlation_groups_path,
    )

    print("Running feature histogram...")
    histogram_path = (
        results_dir
        / f"feature_histogram_{hist_raw_feature_id}_{hist_feature_type.lower()}.png"
    )
    generate_histogram(
        raw_registry_path=base_path / "registries" / "raw_feature",
        agg_registry_path=base_path / "registries" / "agg_feature",
        agg_mapping_path=base_path / "mappings" / "agg_feat_to_raw_feat_mapping",
        raw_feature_id=hist_raw_feature_id,
        feature_type_name=hist_feature_type,
        output_path=histogram_path,
    )

    print("Running executive summary report...")
    executive_summary_path = results_dir / "gait_feature_executive_summary.pdf"
    generate_executive_summary(
        base_path=base_path,
        auc_csv_path=_dated_output_path(auc_summary_path),
        correlation_groups_path=_dated_output_path(correlation_groups_path),
        output_path=executive_summary_path,
    )

    print("\nAll reports complete. Generated artifacts:")
    for expected in [
        dataset_report_path,
        aggregate_report_path,
        auc_summary_path,
        correlation_heatmap_path,
        correlation_groups_path,
        histogram_path,
        executive_summary_path,
    ]:
        print(f"- {_dated_output_path(expected)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all gait-feature visualization reports into one results directory."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base path to converted data (e.g. .../ltmm_2026_02_21).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory where all artifacts are written. "
            "Default: <base-path>/gait_feature_reports"
        ),
    )
    parser.add_argument(
        "--hist-raw-feature-id",
        type=str,
        default=None,
        help=(
            "Raw feature ID for histogram (e.g. raw_<uuid>). "
            "If omitted, auto-selected from registry."
        ),
    )
    parser.add_argument(
        "--hist-feature-type",
        type=str,
        default=None,
        help=(
            "Raw feature enum name for histogram (e.g. STRIDE_TIME). "
            "If omitted, auto-selected from data."
        ),
    )
    parser.add_argument(
        "--corr-min-periods",
        type=int,
        default=25,
        help="Minimum paired samples for correlation computation (default: 25).",
    )
    parser.add_argument(
        "--corr-group-threshold",
        type=float,
        default=0.85,
        help="Absolute correlation threshold for grouping (default: 0.85).",
    )
    parser.add_argument(
        "--corr-annotate",
        action="store_true",
        help="Annotate numeric values on the correlation heatmap.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all_reports(
        base_path=args.base_path,
        results_dir=args.results_dir,
        hist_raw_feature_id=args.hist_raw_feature_id,
        hist_feature_type=args.hist_feature_type,
        corr_min_periods=args.corr_min_periods,
        corr_group_threshold=args.corr_group_threshold,
        corr_annotate=args.corr_annotate,
    )
