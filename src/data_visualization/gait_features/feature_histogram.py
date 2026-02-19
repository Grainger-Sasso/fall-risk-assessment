"""
Script to generate a histogram of raw gait feature values with aggregate
statistics overlaid as vertical reference lines and labels.

The output filename includes the current date (YYYY-MM-DD) before the
extension to keep filenames unique across runs.

Usage:
    python -m src.data_visualization.gait_features.feature_histogram \\
        --raw-registry /path/to/raw_feature_registry \\
        --agg-registry /path/to/agg_feature_registry \\
        --agg-mapping /path/to/agg_feature_mapping \\
        --raw-feature-id raw_<uuid> \\
        --feature-type STRIDE_TIME \\
        [--output /path/to/output.png]
"""

import argparse
from datetime import date
from pathlib import Path
from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.data_io.import_export.importers.features.aggregate.aggregate_feature_importer import (
    AggregateFeatureImporter,
)
from src.data_io.import_export.importers.features.raw.raw_feature_importer import (
    RawFeatureImporter,
)
from src.data_io.import_export.importers.mapping.mapping_importer import MappingImporter
from src.data_io.import_export.importers.registry.registry_importer import (
    RegistryImporter,
)
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier

# Stats to display on the histogram (order and colors)
STAT_DISPLAY_CONFIG = [
    (DescriptiveStatisticType.MEAN, "blue", "mean"),
    (DescriptiveStatisticType.MEDIAN, "green", "median"),
    (DescriptiveStatisticType.STD, "gray", "±1σ"),
    (DescriptiveStatisticType.MIN, "orange", "min"),
    (DescriptiveStatisticType.MAX, "orange", "max"),
    (DescriptiveStatisticType.PERCENTILE_25, "purple", "25th"),
    (DescriptiveStatisticType.PERCENTILE_75, "purple", "75th"),
]


def _extract_raw_values(
    raw_feature_set_entry: RawFeatureSetEntry,
    feature_type: RawFeatureType,
) -> List[float]:
    """Extract raw feature values for the given feature type from all epochs."""
    values = []
    for epoch in raw_feature_set_entry.raw_epoch_features:
        raw_feature = epoch.get_feature_from_type(feature_type)
        if raw_feature is not None and not np.isnan(raw_feature.value):
            values.append(raw_feature.value)
    return values


def _get_aggregate_stats(
    agg_feature_set_entry: AggregateFeatureSetEntry,
    feature_type: RawFeatureType,
) -> dict:
    """Extract aggregate statistics for the given feature type."""
    agg_feature = agg_feature_set_entry.get_feature_from_type(feature_type)
    if agg_feature is None:
        return {}
    return {
        stat.statistic_type: stat.value for stat in agg_feature.descriptive_statistics
    }


def _add_date_to_output_path(output_path: Path) -> Path:
    """Insert current date into filename before extension to make it unique."""
    date_str = date.today().isoformat()
    return output_path.parent / f"{output_path.stem}_{date_str}{output_path.suffix}"


def _find_aggregate_id_from_raw_id(
    mapping_path: Path,
    raw_feature_id: str,
) -> Optional[str]:
    """Find aggregate feature ID that maps to the given raw feature ID."""
    mapping_importer = MappingImporter()
    mapping = mapping_importer.import_data(
        mapping_path,
        AggregateFeatureIdentifier,
        RawFeatureIdentifier,
    )
    for agg_id_value, raw_id_value in mapping.map.items():
        if raw_id_value == raw_feature_id:
            return agg_id_value
    return None


def plot_feature_histogram(
    raw_values: List[float],
    stats: dict,
    feature_type: RawFeatureType,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot histogram of raw values with vertical lines for aggregate statistics.

    Args:
        raw_values: List of raw feature values (one per bout).
        stats: Dict mapping DescriptiveStatisticType to value.
        feature_type: The feature type being plotted (for title).
        output_path: Optional path to save the figure.
    """
    if not raw_values:
        raise ValueError("No raw values to plot")

    fig, ax = plt.subplots(figsize=(10, 6))
    raw_array = np.array(raw_values)

    # Histogram
    n, bins, patches = ax.hist(
        raw_array,
        bins="auto",
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
        label="raw values",
    )

    # Vertical lines for each statistic and collect labels for legend
    legend_entries = []
    for stat_type, color, label in STAT_DISPLAY_CONFIG:
        if stat_type not in stats:
            continue
        value = stats[stat_type]
        if np.isnan(value):
            continue

        if stat_type == DescriptiveStatisticType.STD:
            mean_val = stats.get(DescriptiveStatisticType.MEAN)
            if mean_val is not None and not np.isnan(mean_val):
                ax.axvline(mean_val - value, color=color, linestyle="--", alpha=0.7)
                ax.axvline(mean_val + value, color=color, linestyle="--", alpha=0.7)
                legend_entries.append(
                    (f"μ±σ: [{mean_val - value:.3f}, {mean_val + value:.3f}]", color)
                )
        else:
            ax.axvline(value, color=color, linestyle="--", alpha=0.8)
            legend_entries.append((f"{label}: {value:.3f}", color))

    # Add legend with colored squares for each aggregate stat
    legend_handles = [
        mpatches.Patch(facecolor="steelblue", edgecolor="black", label="raw values")
    ]
    if legend_entries:
        legend_handles.extend(
            [
                mpatches.Patch(facecolor=color, edgecolor="black", label=label_text)
                for label_text, color in legend_entries
            ]
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        framealpha=0.9,
    )

    ax.set_xlabel(feature_type.value)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feature_type.value} (n={len(raw_values)} bouts)")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def main(
    raw_registry_path: Path,
    agg_registry_path: Path,
    agg_mapping_path: Path,
    raw_feature_id: str,
    feature_type_name: str,
    output_path: Optional[Path] = None,
) -> None:
    """
    Load raw and aggregate features from registries and generate histogram.

    Args:
        raw_registry_path: Directory containing raw feature registry.
        agg_registry_path: Directory containing aggregate feature registry.
        agg_mapping_path: Directory containing aggregate->raw mapping.
        raw_feature_id: Raw feature identifier (e.g. raw_<uuid>).
        feature_type_name: RawFeatureType enum name (e.g. STRIDE_TIME).
        output_path: Optional path to save the figure.
    """
    # Resolve feature type
    try:
        feature_type = RawFeatureType[feature_type_name]
    except KeyError:
        raise ValueError(
            f"Unknown feature type: {feature_type_name}. "
            f"Valid options: {[e.name for e in RawFeatureType]}"
        )

    # Load registries
    registry_importer = RegistryImporter()
    raw_registry = registry_importer.import_data(
        raw_registry_path, RawFeatureIdentifier
    )
    agg_registry = registry_importer.import_data(
        agg_registry_path, AggregateFeatureIdentifier
    )

    # Find aggregate ID from mapping
    agg_feature_id = _find_aggregate_id_from_raw_id(agg_mapping_path, raw_feature_id)
    if agg_feature_id is None:
        raise ValueError(
            f"No aggregate feature found for raw feature ID: {raw_feature_id}"
        )

    # Get paths from registries
    raw_id = RawFeatureIdentifier(raw_feature_id)
    agg_id = AggregateFeatureIdentifier(agg_feature_id)
    raw_data_path = raw_registry.get_path_from_id(raw_id)
    agg_data_path = agg_registry.get_path_from_id(agg_id)

    # Load raw and aggregate feature data
    raw_importer = RawFeatureImporter()
    agg_importer = AggregateFeatureImporter()
    raw_feature_set_entry: RawFeatureSetEntry = raw_importer.import_data(raw_data_path)
    agg_feature_set_entry: AggregateFeatureSetEntry = agg_importer.import_data(
        agg_data_path
    )

    # Extract values and stats
    raw_values = _extract_raw_values(raw_feature_set_entry, feature_type)
    stats = _get_aggregate_stats(agg_feature_set_entry, feature_type)

    if not raw_values:
        raise ValueError(
            f"No raw values found for feature type {feature_type_name} "
            f"in raw feature {raw_feature_id}"
        )

    # Add date to output path for uniqueness when saving to file
    if output_path is not None:
        output_path = _add_date_to_output_path(output_path)

    # Plot
    plot_feature_histogram(raw_values, stats, feature_type, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate histogram of raw gait feature values with aggregate statistics."
    )
    parser.add_argument(
        "--raw-registry",
        type=Path,
        required=True,
        help="Path to directory containing raw feature registry (registry.csv)",
    )
    parser.add_argument(
        "--agg-registry",
        type=Path,
        required=True,
        help="Path to directory containing aggregate feature registry (registry.csv)",
    )
    parser.add_argument(
        "--agg-mapping",
        type=Path,
        required=True,
        help="Path to directory containing aggregate->raw mapping (mapping.csv)",
    )
    parser.add_argument(
        "--raw-feature-id",
        type=str,
        required=True,
        help="Raw feature identifier (e.g. raw_<uuid>)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        required=True,
        help="Raw feature type enum name (e.g. STRIDE_TIME, GAIT_SPEED)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the figure; date (YYYY-MM-DD) is appended to filename (default: display interactively)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        raw_registry_path=args.raw_registry,
        agg_registry_path=args.agg_registry,
        agg_mapping_path=args.agg_mapping,
        raw_feature_id=args.raw_feature_id,
        feature_type_name=args.feature_type,
        output_path=args.output,
    )
