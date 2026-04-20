"""
Generate a PDF report on aggregate feature distributions with violin plots.

Aggregate features are descriptive statistics (mean, median, etc.) of raw gait
features computed across epochs from a single IMU recording. This tool plots
only the mean and median aggregate features.

Two report sections:
1. Overall distribution: violin plots of mean/median aggregate features across
   all participants.
2. By class: violin plots of mean/median aggregate features separated by
   faller vs. non-faller.

Uses the same database manager configuration as test_database_validator.

Usage:
    python -m src.data_visualization.gait_features.aggregate_feature_report \\
        --base-path /path/to/converted_data/ltmm_YYYY_MM_DD \\
        [--output /path/to/report.pdf]
"""

import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from src.data_model.data.user.clinical.clinical_demographic_data import (
    FallerStatus,
)
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.database_manager.database_generator import DatabaseGenerator
from src.database_manager.database_manager import DatabaseManager
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier

# Only plot mean and median aggregate statistics
PLOT_STAT_TYPES = [
    DescriptiveStatisticType.MEAN,
    DescriptiveStatisticType.MEDIAN,
]

# Subplots per page: rows x cols
ROWS_PER_PAGE = 3
COLS_PER_PAGE = 2


def _get_database_manager(base_path: Path) -> DatabaseManager:
    """Build database manager using paths from test_database_validator structure."""
    registry_paths = {
        IMUDataIdentifier: base_path / "registries" / "imu_data",
        UserIdentifier: base_path / "registries" / "user_data",
        RawFeatureIdentifier: base_path / "registries" / "raw_feature",
        AggregateFeatureIdentifier: base_path / "registries" / "agg_feature",
    }
    mapping_paths = {
        (IMUDataIdentifier, UserIdentifier): base_path
        / "mappings"
        / "imu_to_user_mapping",
        (RawFeatureIdentifier, IMUDataIdentifier): base_path
        / "mappings"
        / "raw_feat_to_imu_mapping",
        (AggregateFeatureIdentifier, RawFeatureIdentifier): base_path
        / "mappings"
        / "agg_feat_to_raw_feat_mapping",
    }
    output_dir_paths = {
        RawFeatureIdentifier: base_path / "raw_feat",
        AggregateFeatureIdentifier: base_path / "agg_feat",
    }
    db_generator = DatabaseGenerator()
    return db_generator.generate_database(
        registry_paths, mapping_paths, output_dir_paths, validate=False
    )


def _load_aggregate_feature_data(
    db_manager: DatabaseManager,
) -> Tuple[
    Dict[Tuple[RawFeatureType, DescriptiveStatisticType], List[float]],
    Dict[Tuple[RawFeatureType, DescriptiveStatisticType], Dict[FallerStatus, List[float]]],
]:
    """
    Load aggregate features and extract mean/median values per participant.

    Returns:
        overall: Dict mapping (feature_type, stat_type) -> list of values (all participants)
        by_class: Dict mapping (feature_type, stat_type) -> {FallerStatus: list of values}
    """
    agg_registry = db_manager.registry_manager.get_provider(
        AggregateFeatureIdentifier
    )
    overall: Dict[
        Tuple[RawFeatureType, DescriptiveStatisticType], List[float]
    ] = defaultdict(list)
    by_class: Dict[
        Tuple[RawFeatureType, DescriptiveStatisticType],
        Dict[FallerStatus, List[float]],
    ] = defaultdict(lambda: {FallerStatus.FALLER: [], FallerStatus.NON_FALLER: []})

    for agg_feature_id in agg_registry.registry.keys():
        agg_entry: AggregateFeatureSetEntry = db_manager.import_data(
            [AggregateFeatureIdentifier(agg_feature_id)]
        )[0]
        user_id = agg_entry.metadata.user_identifier
        user_data = db_manager.import_data([user_id])[0]
        faller_status = user_data.clinical_demographic_data.faller_status

        for agg_feature in agg_entry.aggregate_features:
            feature_type = agg_feature.feature_type
            for stat_type in PLOT_STAT_TYPES:
                stat = agg_feature.get_statistic_from_type(stat_type)
                if stat is None:
                    continue
                value = stat.value
                if np.isnan(value):
                    continue

                key = (feature_type, stat_type)
                overall[key].append(value)
                if faller_status in (FallerStatus.FALLER, FallerStatus.NON_FALLER):
                    by_class[key][faller_status].append(value)

    return dict(overall), dict(by_class)


def _add_date_to_output_path(output_path: Path) -> Path:
    """Insert current date into filename before extension."""
    date_str = date.today().isoformat()
    return output_path.parent / f"{output_path.stem}_{date_str}{output_path.suffix}"


def _compute_auc_roc_from_groups(
    faller_values: List[float],
    non_faller_values: List[float],
) -> Optional[float]:
    """
    Compute AUC-ROC for a single feature from two score groups.

    Uses the probabilistic interpretation:
        AUC = P(score_faller > score_non_faller) + 0.5 * P(tie)
    """
    if not faller_values or not non_faller_values:
        return None

    faller = np.asarray(faller_values, dtype=float)
    non_faller = np.asarray(non_faller_values, dtype=float)
    total_pairs = float(faller.size * non_faller.size)
    if total_pairs == 0:
        return None

    wins = 0.0
    ties = 0.0
    for score in faller:
        wins += float(np.sum(score > non_faller))
        ties += float(np.sum(score == non_faller))

    return (wins + 0.5 * ties) / total_pairs


def _format_auc(auc: Optional[float]) -> str:
    """Format AUC value for display."""
    if auc is None:
        return "AUC=N/A"
    return f"AUC={auc:.3f}"


def _build_auc_summary_rows(
    by_class: Dict[
        Tuple[RawFeatureType, DescriptiveStatisticType],
        Dict[FallerStatus, List[float]],
    ],
) -> List[Dict[str, str]]:
    """Create row records for per-feature AUC summary export."""
    rows: List[Dict[str, str]] = []
    keys = sorted(by_class.keys(), key=lambda k: (k[0].value, k[1].value))

    for key in keys:
        feature_type, stat_type = key
        faller_vals = by_class[key][FallerStatus.FALLER]
        non_faller_vals = by_class[key][FallerStatus.NON_FALLER]
        auc = _compute_auc_roc_from_groups(faller_vals, non_faller_vals)
        rows.append(
            {
                "feature_type": feature_type.value,
                "stat_type": stat_type.value,
                "n_faller": str(len(faller_vals)),
                "n_non_faller": str(len(non_faller_vals)),
                "auc_roc": "" if auc is None else f"{auc:.6f}",
            }
        )
    return rows


def _write_auc_summary_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Write per-feature AUC summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["feature_type", "stat_type", "n_faller", "n_non_faller", "auc_roc"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved AUC summary to {output_path}")


def _create_overall_violin_plots(
    overall: Dict[Tuple[RawFeatureType, DescriptiveStatisticType], List[float]],
    pdf: PdfPages,
) -> None:
    """Create violin plots for mean/median aggregate features (all participants)."""
    keys = sorted(overall.keys(), key=lambda k: (k[0].value, k[1].value))
    if not keys:
        return

    n_total = len(keys)
    n_per_page = ROWS_PER_PAGE * COLS_PER_PAGE
    n_pages = (n_total + n_per_page - 1) // n_per_page

    for page in range(n_pages):
        start = page * n_per_page
        end = min(start + n_per_page, n_total)
        page_keys = keys[start:end]

        fig, axes = plt.subplots(
            ROWS_PER_PAGE, COLS_PER_PAGE, figsize=(10, 12), squeeze=False
        )
        axes_flat = axes.flatten()

        for idx, key in enumerate(page_keys):
            ax = axes_flat[idx]
            feature_type, stat_type = key
            values = overall[key]
            if not values:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                parts = ax.violinplot(
                    [values],
                    positions=[0],
                    showmeans=True,
                    showmedians=True,
                    widths=0.7,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor("steelblue")
                    pc.set_alpha(0.8)
                ax.set_xticks([0])
                ax.set_xticklabels([f"{feature_type.value}\n({stat_type.value})"])
                ax.set_ylabel("Value")
                ax.set_title(f"n={len(values)}")

            ax.set_xlim(-0.5, 0.5)

        for idx in range(len(page_keys), n_per_page):
            axes_flat[idx].axis("off")

        fig.suptitle(
            f"Aggregate Feature Distributions (Mean/Median) — Page {page + 1}/{n_pages}",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close()


def _create_class_separated_violin_plots(
    by_class: Dict[
        Tuple[RawFeatureType, DescriptiveStatisticType],
        Dict[FallerStatus, List[float]],
    ],
    pdf: PdfPages,
) -> None:
    """Create violin plots for mean/median aggregate features by faller status."""
    keys = sorted(by_class.keys(), key=lambda k: (k[0].value, k[1].value))
    keys = [
        k
        for k in keys
        if by_class[k][FallerStatus.FALLER] or by_class[k][FallerStatus.NON_FALLER]
    ]
    if not keys:
        return

    n_total = len(keys)
    n_per_page = ROWS_PER_PAGE * COLS_PER_PAGE
    n_pages = (n_total + n_per_page - 1) // n_per_page

    for page in range(n_pages):
        start = page * n_per_page
        end = min(start + n_per_page, n_total)
        page_keys = keys[start:end]

        fig, axes = plt.subplots(
            ROWS_PER_PAGE, COLS_PER_PAGE, figsize=(10, 12), squeeze=False
        )
        axes_flat = axes.flatten()

        for idx, key in enumerate(page_keys):
            ax = axes_flat[idx]
            feature_type, stat_type = key
            faller_vals = by_class[key][FallerStatus.FALLER]
            non_faller_vals = by_class[key][FallerStatus.NON_FALLER]
            auc = _compute_auc_roc_from_groups(faller_vals, non_faller_vals)

            class_data = [
                ("Faller", faller_vals, "#e74c3c"),
                ("Non-faller", non_faller_vals, "#3498db"),
            ]
            class_data = [entry for entry in class_data if entry[1]]

            data = [values for _, values, _ in class_data]
            if not data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                positions = list(range(len(data)))
                parts = ax.violinplot(
                    data,
                    positions=positions,
                    showmeans=True,
                    showmedians=True,
                    widths=0.7,
                )
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(class_data[i][2])
                    pc.set_alpha(0.8)
                ax.set_xticks(positions)
                ax.set_xticklabels(
                    [label for label, _, _ in class_data],
                    fontsize=9,
                )
                ax.set_ylabel("Value")
                ax.set_title(
                    f"{feature_type.value} ({stat_type.value})\n"
                    f"n_faller={len(faller_vals)}, n_non_faller={len(non_faller_vals)}, "
                    f"{_format_auc(auc)}"
                )

        for idx in range(len(page_keys), n_per_page):
            axes_flat[idx].axis("off")

        fig.suptitle(
            f"Aggregate Feature Distributions by Class — Page {page + 1}/{n_pages}",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close()


def generate_report(
    base_path: Path,
    output_path: Optional[Path] = None,
    auc_output_path: Optional[Path] = None,
) -> None:
    """
    Generate the aggregate feature distribution report PDF.

    Args:
        base_path: Base path to converted data (e.g. .../ltmm_2026_02_21).
        output_path: Where to save the PDF. Date is appended to filename.
        auc_output_path: Where to save per-feature AUC summary CSV.
    """
    db_manager = _get_database_manager(base_path)
    overall, by_class = _load_aggregate_feature_data(db_manager)

    if output_path is None:
        output_path = base_path / "aggregate_feature_report.pdf"
    output_path = _add_date_to_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        _create_overall_violin_plots(overall, pdf)
        _create_class_separated_violin_plots(by_class, pdf)

    auc_rows = _build_auc_summary_rows(by_class)
    if auc_output_path is None:
        auc_output_path = base_path / "aggregate_feature_auc_summary.csv"
    auc_output_path = _add_date_to_output_path(auc_output_path)
    _write_auc_summary_csv(auc_rows, auc_output_path)

    print(f"Saved report to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDF report on aggregate feature distributions (mean/median) with violin plots."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base path to converted data (e.g. ltmm_2026_02_21)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <base-path>/aggregate_feature_report_<date>.pdf)",
    )
    parser.add_argument(
        "--auc-output",
        type=Path,
        default=None,
        help="AUC CSV path (default: <base-path>/aggregate_feature_auc_summary_<date>.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_report(
        base_path=args.base_path,
        output_path=args.output,
        auc_output_path=args.auc_output,
    )
