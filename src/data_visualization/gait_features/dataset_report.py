"""
Generate a PDF report on class representation (fallers vs non-fallers) and
feature missingness for raw gait features.

Raw features are structured as N participants (unique users) × M epochs per
participant, where M varies. The report includes:
- Class representation: count of faller vs non-faller participants
- Feature count by class: number of epochs (features) belonging to fallers vs
  non-fallers
- Feature missingness: bar plot of percentage missing by feature type

Uses the same database manager configuration as test_database_validator.

Usage:
    python -m src.data_visualization.gait_features.dataset_report \\
        --base-path /path/to/converted_data/ltmm_YYYY_MM_DD \\
        [--output /path/to/report.pdf]
"""

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data_model.data.user.clinical.clinical_demographic_data import (
    FallerStatus,
)
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.database_manager.database_generator import DatabaseGenerator
from src.database_manager.database_manager import DatabaseManager
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


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


def _compute_class_representation(
    db_manager: DatabaseManager,
) -> Tuple[int, int, int, int, int, int]:
    """
    Compute faller vs non-faller counts from raw features.

    Returns:
        Tuple of (n_fallers, n_non_fallers, n_unknown,
                  n_epochs_fallers, n_epochs_non_fallers, n_epochs_unknown).
    """
    raw_registry = db_manager.registry_manager.get_provider(RawFeatureIdentifier)
    seen_users: Set[str] = set()
    fallers = 0
    non_fallers = 0
    unknown = 0
    epochs_fallers = 0
    epochs_non_fallers = 0
    epochs_unknown = 0

    for raw_feature_id in raw_registry.registry.keys():
        raw_feature_set: RawFeatureSetEntry = db_manager.import_data(
            [RawFeatureIdentifier(raw_feature_id)]
        )[0]
        user_id = raw_feature_set.metadata.user_identifier
        n_epochs = len(raw_feature_set.raw_epoch_features)

        user_data = db_manager.import_data([user_id])[0]
        status = user_data.clinical_demographic_data.faller_status
        is_new_user = user_id.value not in seen_users
        seen_users.add(user_id.value)

        if status == FallerStatus.FALLER:
            fallers += 1 if is_new_user else 0
            epochs_fallers += n_epochs
        elif status == FallerStatus.NON_FALLER:
            non_fallers += 1 if is_new_user else 0
            epochs_non_fallers += n_epochs
        else:
            unknown += 1 if is_new_user else 0
            epochs_unknown += n_epochs

    return fallers, non_fallers, unknown, epochs_fallers, epochs_non_fallers, epochs_unknown


def _compute_feature_missingness(
    db_manager: DatabaseManager,
) -> Dict[RawFeatureType, float]:
    """
    Compute percentage of missing values per feature type across all epochs.

    Missing = feature not present in epoch OR value is NaN.
    Total cells per feature type = total epochs (N participants × M epochs).

    Returns:
        Dict mapping RawFeatureType to percentage missing (0-100).
    """
    raw_registry = db_manager.registry_manager.get_provider(RawFeatureIdentifier)

    # First pass: collect all feature types and total epochs
    all_feature_types: Set[RawFeatureType] = set()
    total_epochs = 0
    for raw_feature_id in raw_registry.registry.keys():
        raw_feature_set = db_manager.import_data(
            [RawFeatureIdentifier(raw_feature_id)]
        )[0]
        total_epochs += len(raw_feature_set.raw_epoch_features)
        for epoch in raw_feature_set.raw_epoch_features:
            for raw_feature in epoch.raw_features:
                all_feature_types.add(raw_feature.feature_type)

    # Second pass: count missing (absent or NaN) per feature type
    missing_by_type: Dict[RawFeatureType, int] = defaultdict(int)
    for raw_feature_id in raw_registry.registry.keys():
        raw_feature_set = db_manager.import_data(
            [RawFeatureIdentifier(raw_feature_id)]
        )[0]
        for epoch in raw_feature_set.raw_epoch_features:
            for feat_type in all_feature_types:
                raw_feature = epoch.get_feature_from_type(feat_type)
                if raw_feature is None or np.isnan(raw_feature.value):
                    missing_by_type[feat_type] += 1

    pct_missing: Dict[RawFeatureType, float] = {}
    for feat_type in all_feature_types:
        missing = missing_by_type[feat_type]
        pct_missing[feat_type] = (
            (missing / total_epochs * 100) if total_epochs > 0 else 0.0
        )

    return pct_missing


def _add_date_to_output_path(output_path: Path) -> Path:
    """Insert current date into filename before extension."""
    date_str = date.today().isoformat()
    return output_path.parent / f"{output_path.stem}_{date_str}{output_path.suffix}"


def _create_report_pdf(
    n_fallers: int,
    n_non_fallers: int,
    n_unknown: int,
    n_epochs_fallers: int,
    n_epochs_non_fallers: int,
    n_epochs_unknown: int,
    pct_missing: Dict[RawFeatureType, float],
    output_path: Path,
) -> None:
    """Create a PDF report with class representation and missingness bar plot."""
    fig = plt.figure(figsize=(11, 8.5))

    # Class representation text
    ax_text = fig.add_axes([0.1, 0.6, 0.8, 0.35])
    ax_text.axis("off")
    total_labeled = n_fallers + n_non_fallers
    total_epochs = n_epochs_fallers + n_epochs_non_fallers + n_epochs_unknown
    text = (
        "Class Representation (Participants)\n\n"
        f"Fallers: {n_fallers}\n"
        f"Non-fallers: {n_non_fallers}\n"
        f"Total (labeled): {total_labeled}\n"
    )
    if n_unknown > 0:
        text += f"Unknown: {n_unknown}\n"
    text += (
        "\nFeature Count by Class (Epochs)\n\n"
        f"Fallers: {n_epochs_fallers} epochs\n"
        f"Non-fallers: {n_epochs_non_fallers} epochs\n"
        f"Total: {total_epochs} epochs\n"
    )
    if n_epochs_unknown > 0:
        text += f"Unknown: {n_epochs_unknown} epochs\n"
    ax_text.text(0.5, 0.5, text, fontsize=12, verticalalignment="center")

    # Bar plot of missingness
    ax_bar = fig.add_axes([0.1, 0.08, 0.85, 0.48])
    sorted_items = sorted(
        pct_missing.items(), key=lambda x: x[1], reverse=True
    )
    feature_labels = [ft.value for ft, _ in sorted_items]
    pct_values = [pct for _, pct in sorted_items]

    x_pos = np.arange(len(feature_labels))
    ax_bar.barh(x_pos, pct_values, color="steelblue", alpha=0.8)
    ax_bar.set_yticks(x_pos)
    ax_bar.set_yticklabels(feature_labels, fontsize=8)
    ax_bar.set_xlabel("Percentage Missing (%)")
    ax_bar.set_title("Feature Missingness by Feature Type")
    ax_bar.invert_yaxis()

    plt.suptitle("Dataset Exploratory Report", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved report to {output_path}")


def generate_report(
    base_path: Path,
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate the dataset report PDF.

    Args:
        base_path: Base path to converted data (e.g. .../ltmm_2026_02_21).
        output_path: Where to save the PDF. Date is appended to filename.
    """
    db_manager = _get_database_manager(base_path)

    (
        n_fallers,
        n_non_fallers,
        n_unknown,
        n_epochs_fallers,
        n_epochs_non_fallers,
        n_epochs_unknown,
    ) = _compute_class_representation(db_manager)
    pct_missing = _compute_feature_missingness(db_manager)

    if output_path is None:
        output_path = base_path / "dataset_report.pdf"
    output_path = _add_date_to_output_path(output_path)

    _create_report_pdf(
        n_fallers,
        n_non_fallers,
        n_unknown,
        n_epochs_fallers,
        n_epochs_non_fallers,
        n_epochs_unknown,
        pct_missing,
        output_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDF report on class representation and feature missingness."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base path to converted data (e.g. .../ltmm_2026_02_21)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <base-path>/dataset_report_<date>.pdf)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_report(base_path=args.base_path, output_path=args.output)
