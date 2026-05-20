"""
Generate a feature correlation heatmap from raw gait feature epochs.

This tool loads all raw epoch-level features from a converted data directory,
builds a feature matrix (epochs x feature types), computes pairwise
correlations, and saves a heatmap image.

Usage:
    python -m src.data_visualization.gait_features.feature_correlation_heatmap \
        --base-path /path/to/converted_data/ltmm_YYYY_MM_DD \
        [--output /path/to/heatmap.png] \
        [--group-threshold 0.85] \
        [--groups-output /path/to/groups.txt]
"""

import argparse
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.database_manager.database_generator import DatabaseGenerator
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


def _get_database_manager(base_path: Path) -> DatabaseManager:
    """Build database manager from converted data directories."""
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
        RawFeatureIdentifier: base_path / "raw_feature",
        AggregateFeatureIdentifier: base_path / "agg_feature",
    }
    db_generator = DatabaseGenerator()
    return db_generator.generate_database(
        registry_paths, mapping_paths, output_dir_paths, validate=False
    )


def _extract_epoch_feature_rows(db_manager: DatabaseManager) -> List[Dict[str, float]]:
    """
    Build epoch-level feature rows from all raw feature entries.

    Returns:
        A list of row dictionaries where keys are feature names and values
        are numeric feature values.
    """
    raw_registry = db_manager.registry_manager.get_provider(RawFeatureIdentifier)
    rows: List[Dict[str, float]] = []

    for raw_feature_id in raw_registry.registry.keys():
        raw_feature_set = db_manager.import_data(
            [RawFeatureIdentifier(raw_feature_id)]
        )[0]

        for epoch in raw_feature_set.raw_epoch_features:
            row: Dict[str, float] = {}
            for raw_feature in epoch.raw_features:
                value = raw_feature.value
                if value is None:
                    continue
                if not isinstance(value, (int, float, np.number)):
                    continue
                if np.isnan(value):
                    continue
                row[raw_feature.feature_type.value] = float(value)

            if row:
                rows.append(row)

    return rows


def _build_feature_matrix(
    rows: List[Dict[str, float]],
    min_periods: int,
) -> Tuple[np.ndarray, List[str]]:
    """Create an epoch x feature matrix after filtering sparse/constant features."""
    if not rows:
        raise ValueError("No valid epoch feature values found to compute correlations.")

    feature_names = sorted({feature for row in rows for feature in row.keys()})
    if not feature_names:
        raise ValueError("No feature columns found to compute correlations.")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    matrix = np.full((len(rows), len(feature_names)), np.nan, dtype=float)
    for row_idx, row in enumerate(rows):
        for feature_name, value in row.items():
            matrix[row_idx, name_to_idx[feature_name]] = value

    valid_counts = np.sum(~np.isnan(matrix), axis=0)
    kept_mask = valid_counts >= min_periods
    if not np.any(kept_mask):
        raise ValueError(
            "No features have enough valid samples. "
            f"Try lowering --min-periods (currently {min_periods})."
        )
    matrix = matrix[:, kept_mask]
    kept_names = [name for name, keep in zip(feature_names, kept_mask) if keep]

    non_constant_mask = np.array(
        [
            np.unique(col[~np.isnan(col)]).size > 1
            for col in matrix.T
        ],
        dtype=bool,
    )
    if np.sum(non_constant_mask) < 2:
        raise ValueError(
            "Need at least two non-constant features to compute correlations."
        )
    matrix = matrix[:, non_constant_mask]
    final_names = [name for name, keep in zip(kept_names, non_constant_mask) if keep]
    return matrix, final_names


def _compute_pairwise_pearson_correlation(
    matrix: np.ndarray,
    min_periods: int,
) -> np.ndarray:
    """Compute pairwise-complete Pearson correlations for matrix columns."""
    n_features = matrix.shape[1]
    corr_matrix = np.full((n_features, n_features), np.nan, dtype=float)

    for i in range(n_features):
        xi = matrix[:, i]
        for j in range(i, n_features):
            xj = matrix[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            n_overlap = int(np.sum(mask))
            if n_overlap < min_periods:
                continue

            xi_valid = xi[mask]
            xj_valid = xj[mask]
            xi_std = np.std(xi_valid)
            xj_std = np.std(xj_valid)
            if xi_std == 0 or xj_std == 0:
                continue

            corr = float(np.corrcoef(xi_valid, xj_valid)[0, 1])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def _group_highly_correlated_features(
    corr_matrix: np.ndarray,
    feature_names: Sequence[str],
    threshold: float,
) -> List[List[str]]:
    """
    Group features into connected components where an edge exists when
    abs(correlation) > threshold.
    """
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if corr_matrix.shape[0] != len(feature_names):
        raise ValueError("feature_names must align with corr_matrix dimensions.")

    n = len(feature_names)
    adjacency: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = corr_matrix[i, j]
            if np.isnan(value):
                continue
            if abs(float(value)) > threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = [False] * n
    groups: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        if not adjacency[start]:
            visited[start] = True
            continue  # ignore singletons

        stack = [start]
        visited[start] = True
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in adjacency[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        component.sort(key=lambda idx: feature_names[idx])
        groups.append(component)

    # Largest groups first, then alphabetical by first feature for stability
    groups.sort(
        key=lambda comp: (-len(comp), feature_names[comp[0]] if comp else "")
    )
    return [[feature_names[i] for i in comp] for comp in groups]


def _format_groups_text(groups: List[List[str]], threshold: float) -> str:
    lines: List[str] = []
    lines.append(f"Highly-correlated feature groups (abs(r) > {threshold:.2f})")
    lines.append("")
    if not groups:
        lines.append("(no groups found above threshold)")
        lines.append("")
        return "\n".join(lines)

    for idx, group in enumerate(groups, start=1):
        lines.append(f"Group {idx} (n={len(group)}):")
        for name in group:
            lines.append(f"  - {name}")
        lines.append("")
    return "\n".join(lines)


def _add_date_to_output_path(output_path: Path) -> Path:
    """Insert current date into filename before extension."""
    date_str = date.today().isoformat()
    return output_path.parent / f"{output_path.stem}_{date_str}{output_path.suffix}"


def _plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: Sequence[str],
    output_path: Path,
    annotate: bool,
) -> None:
    """Plot and save a feature correlation heatmap."""
    n_features = len(feature_names)
    fig_size = max(8, min(26, 0.45 * n_features))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cmap = plt.cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(
        corr_matrix,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title(f"Feature Correlation Heatmap (n={n_features} features)")

    if annotate and n_features <= 50:
        for i in range(n_features):
            for j in range(n_features):
                value = corr_matrix[i, j]
                if np.isnan(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation coefficient")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation heatmap to {output_path}")


def generate_feature_correlation_heatmap(
    base_path: Path,
    output_path: Optional[Path] = None,
    min_periods: int = 25,
    annotate: bool = False,
    group_threshold: float = 0.85,
    groups_output_path: Optional[Path] = None,
) -> Path:
    """
    Generate and save a raw-feature correlation heatmap image.

    Args:
        base_path: Base path to converted data (e.g. .../ltmm_2026_02_21).
        output_path: Output image path (date appended automatically).
        min_periods: Minimum overlapping samples required for each pairwise
            correlation and minimum valid values for a feature to be retained.
        annotate: Whether to print correlation values inside heatmap cells.
        group_threshold: Threshold used to form groups of highly correlated
            features using abs(r) > threshold.
        groups_output_path: Optional path to save the groups text output
            (date appended automatically).

    Returns:
        Final output path where image was saved.
    """
    db_manager = _get_database_manager(base_path)
    rows = _extract_epoch_feature_rows(db_manager)
    feature_matrix, feature_names = _build_feature_matrix(rows, min_periods=min_periods)
    corr_matrix = _compute_pairwise_pearson_correlation(
        feature_matrix,
        min_periods=min_periods,
    )

    groups = _group_highly_correlated_features(
        corr_matrix=corr_matrix,
        feature_names=feature_names,
        threshold=group_threshold,
    )
    groups_text = _format_groups_text(groups, threshold=group_threshold)
    print(groups_text)
    if groups_output_path is None:
        groups_output_path = base_path / "feature_correlation_groups.txt"
    groups_output_path = _add_date_to_output_path(groups_output_path)
    groups_output_path.parent.mkdir(parents=True, exist_ok=True)
    groups_output_path.write_text(groups_text, encoding="utf-8")
    print(f"Saved correlation groups to {groups_output_path}")

    if output_path is None:
        output_path = base_path / "feature_correlation_heatmap.png"
    output_path = _add_date_to_output_path(output_path)

    _plot_correlation_heatmap(
        corr_matrix=corr_matrix,
        feature_names=feature_names,
        output_path=output_path,
        annotate=annotate,
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a feature correlation heatmap from raw gait features."
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
        help="Output image path (default: <base-path>/feature_correlation_heatmap_<date>.png)",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=25,
        help="Minimum valid paired samples for each correlation (default: 25)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate cells with numeric correlation values (auto-disabled above 50 features).",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.85,
        help="Threshold for grouping features by abs(r) > threshold (default: 0.85)",
    )
    parser.add_argument(
        "--groups-output",
        type=Path,
        default=None,
        help="Optional path to save groups text (default: <base-path>/feature_correlation_groups_<date>.txt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_feature_correlation_heatmap(
        base_path=args.base_path,
        output_path=args.output,
        min_periods=args.min_periods,
        annotate=args.annotate,
        group_threshold=args.group_threshold,
        groups_output_path=args.groups_output,
    )
