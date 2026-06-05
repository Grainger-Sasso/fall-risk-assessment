from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis


@dataclass
class FeaturePlotEngine:
    figure: Figure

    def render_class_violin(
        self,
        grouped_values: Dict[str, np.ndarray],
        feature_type: FeatureType,
        basis: SampleBasis,
    ) -> Dict[str, Dict[str, float]]:
        self.figure.clear()
        ax_violin = self.figure.add_subplot(211)
        ax_hist = self.figure.add_subplot(212)

        if not grouped_values:
            ax_violin.text(0.5, 0.5, "No feature values available", ha="center", va="center")
            ax_violin.set_axis_off()
            ax_hist.set_axis_off()
            self.figure.tight_layout()
            return {}

        class_labels = sorted(grouped_values.keys())
        values = [grouped_values[label] for label in class_labels]
        violin_parts = ax_violin.violinplot(
            values,
            showmeans=True,
            showmedians=True,
            widths=0.7,
        )
        for body in violin_parts["bodies"]:
            body.set_alpha(0.35)

        ax_violin.set_xticks(range(1, len(class_labels) + 1))
        ax_violin.set_xticklabels(class_labels, rotation=0)
        ax_violin.set_ylabel("Feature value")
        ax_violin.set_title(f"{feature_type.value} ({basis.value}) by class")
        ax_violin.grid(alpha=0.2)

        for class_label in class_labels:
            ax_hist.hist(
                grouped_values[class_label],
                bins=40,
                alpha=0.35,
                density=True,
                label=class_label,
            )
        ax_hist.set_xlabel("Feature value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Distribution by class")
        ax_hist.grid(alpha=0.2)
        ax_hist.legend(loc="upper right")

        self.figure.tight_layout()
        return self._compute_summary(grouped_values)

    @staticmethod
    def _compute_summary(grouped_values: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for class_label, values in grouped_values.items():
            if values.size == 0:
                continue
            summary[class_label] = {
                "count": float(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            }
        return summary

    # ------------------------------------------------------------------
    # Population-level analytics
    # ------------------------------------------------------------------
    def render_feature_correlation_heatmap(
        self,
        feature_types: List[FeatureType],
        feature_matrix: np.ndarray,
        basis: SampleBasis,
        title: Optional[str] = None,
    ) -> np.ndarray:
        """Render a Pearson correlation heatmap across feature types."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        matrix = np.asarray(feature_matrix, dtype=float)
        correlation = self.compute_correlation_matrix(matrix)

        if correlation.size == 0 or not feature_types:
            ax.text(0.5, 0.5, "No feature values available", ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            return correlation

        image = ax.imshow(
            np.ma.masked_invalid(correlation),
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            aspect="auto",
        )
        labels = [feature_type.value for feature_type in feature_types]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title or f"Feature correlation ({basis.value})")
        self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
        self.figure.tight_layout()
        return correlation

    def render_class_separation(
        self,
        feature_matrix: np.ndarray,
        row_class_labels: List[str],
        feature_types: List[FeatureType],
        basis: SampleBasis,
        top_n: Optional[int] = None,
    ) -> Dict[FeatureType, float]:
        """Render Cohen's d per feature between the two classes (ranked)."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        separation, classes = self.compute_class_separation(
            np.asarray(feature_matrix, dtype=float), row_class_labels, feature_types
        )

        if not separation:
            ax.text(
                0.5,
                0.5,
                "Class separation requires exactly two populated classes",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        ordered = sorted(separation.items(), key=lambda kv: abs(kv[1]), reverse=True)
        if top_n is not None and top_n > 0:
            ordered = ordered[:top_n]
        names = [feature_type.value for feature_type, _ in ordered]
        values = [value for _, value in ordered]
        positions = range(len(ordered))

        ax.barh(list(positions), values, color="#4c72b0")
        ax.set_yticks(list(positions))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(f"Cohen's d  ({classes[0]} minus {classes[1]})")
        ax.set_title(f"Class separation by feature ({basis.value})")
        ax.grid(alpha=0.2, axis="x")
        self.figure.tight_layout()
        return dict(ordered)

    def render_feature_coverage(
        self,
        feature_matrix: np.ndarray,
        row_class_labels: List[str],
        feature_types: List[FeatureType],
        basis: SampleBasis,
    ) -> Dict[FeatureType, Dict[str, float]]:
        """Render the fraction of valid (non-NaN) samples per feature."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        coverage = self.compute_feature_coverage(
            np.asarray(feature_matrix, dtype=float), row_class_labels, feature_types
        )

        if not coverage:
            ax.text(0.5, 0.5, "No feature values available", ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        items = list(coverage.items())
        names = [feature_type.value for feature_type, _ in items]
        fractions = [stats["valid_fraction"] for _, stats in items]
        positions = range(len(items))
        colors = [
            "#d62728" if fraction == 0.0 else ("#ff7f0e" if fraction < 0.5 else "#2ca02c")
            for fraction in fractions
        ]

        ax.barh(list(positions), fractions, color=colors)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Valid sample fraction")
        ax.set_title(f"Feature coverage ({basis.value})")
        ax.grid(alpha=0.2, axis="x")
        self.figure.tight_layout()
        return coverage

    def render_sample_count_by_basis_class(
        self,
        counts_by_basis_class: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Grouped bar chart of mean +/- std per-participant sample count, with the
        x-axis split by class and one bar per sampling basis within each class.

        ``counts_by_basis_class`` is ``{basis_value: {class_label: counts}}`` as
        produced by ``VisualizationDataService.collect_sample_counts_by_basis_class``.
        Returns a summary keyed by ``"basis|class"``.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        bases = [SampleBasis.EPOCH.value, SampleBasis.STRIDE.value]
        classes = sorted(
            {
                class_label
                for basis in bases
                for class_label in counts_by_basis_class.get(basis, {})
            }
        )

        if not classes:
            ax.text(0.5, 0.5, "No samples available", ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        summary: Dict[str, Dict[str, float]] = {}
        bar_colors = {
            SampleBasis.EPOCH.value: "#4c72b0",
            SampleBasis.STRIDE.value: "#dd8452",
        }
        width = 0.38
        x = np.arange(len(classes))

        for basis_index, basis in enumerate(bases):
            means: List[float] = []
            stds: List[float] = []
            for class_label in classes:
                counts = counts_by_basis_class.get(basis, {}).get(
                    class_label, np.array([], dtype=float)
                )
                mean = float(np.mean(counts)) if counts.size else 0.0
                std = float(np.std(counts)) if counts.size else 0.0
                means.append(mean)
                stds.append(std)
                summary[f"{basis}|{class_label}"] = {
                    "mean": mean,
                    "std": std,
                    "n_participants": float(counts.size),
                }
            offset = (basis_index - (len(bases) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=basis,
                color=bar_colors.get(basis),
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(classes)
        ax.set_xlabel("Class")
        ax.set_ylabel("Mean samples per participant")
        ax.set_title("Per-participant sample count by class and basis (mean +/- std)")
        ax.legend(title="Sample basis")
        ax.grid(alpha=0.2, axis="y")
        self.figure.tight_layout()
        return summary

    @staticmethod
    def compute_correlation_matrix(feature_matrix: np.ndarray) -> np.ndarray:
        """Pairwise-complete Pearson correlation across feature columns."""
        matrix = np.asarray(feature_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            return np.empty((0, 0))
        num_features = matrix.shape[1]
        correlation = np.full((num_features, num_features), np.nan)
        for i in range(num_features):
            for j in range(num_features):
                left = matrix[:, i]
                right = matrix[:, j]
                mask = np.isfinite(left) & np.isfinite(right)
                if int(mask.sum()) < 3:
                    continue
                left_valid = left[mask]
                right_valid = right[mask]
                if left_valid.std() < 1e-12 or right_valid.std() < 1e-12:
                    continue
                correlation[i, j] = float(np.corrcoef(left_valid, right_valid)[0, 1])
        return correlation

    @staticmethod
    def compute_class_separation(
        feature_matrix: np.ndarray,
        row_class_labels: List[str],
        feature_types: List[FeatureType],
    ) -> Tuple[Dict[FeatureType, float], List[str]]:
        """Cohen's d per feature between exactly two classes."""
        classes = sorted(set(row_class_labels))
        if len(classes) != 2 or feature_matrix.size == 0:
            return {}, classes
        labels = np.asarray(row_class_labels)
        first_mask = labels == classes[0]
        second_mask = labels == classes[1]
        separation: Dict[FeatureType, float] = {}
        for col, feature_type in enumerate(feature_types):
            first = feature_matrix[first_mask, col]
            second = feature_matrix[second_mask, col]
            first = first[np.isfinite(first)]
            second = second[np.isfinite(second)]
            if first.size < 2 or second.size < 2:
                continue
            pooled_variance = (
                (first.size - 1) * first.var(ddof=1)
                + (second.size - 1) * second.var(ddof=1)
            ) / (first.size + second.size - 2)
            pooled_std = float(np.sqrt(pooled_variance))
            if pooled_std < 1e-12:
                continue
            separation[feature_type] = float((first.mean() - second.mean()) / pooled_std)
        return separation, classes

    @staticmethod
    def compute_feature_coverage(
        feature_matrix: np.ndarray,
        row_class_labels: List[str],
        feature_types: List[FeatureType],
    ) -> Dict[FeatureType, Dict[str, float]]:
        """Per-feature valid-sample counts and fractions (overall and per class)."""
        if feature_matrix.ndim != 2 or feature_matrix.shape[1] == 0:
            return {}
        total_rows = feature_matrix.shape[0]
        labels = np.asarray(row_class_labels)
        classes = sorted(set(row_class_labels))
        coverage: Dict[FeatureType, Dict[str, float]] = {}
        for col, feature_type in enumerate(feature_types):
            column = feature_matrix[:, col]
            valid = int(np.isfinite(column).sum())
            per_class: Dict[str, Dict[str, int]] = {}
            for class_label in classes:
                class_mask = labels == class_label
                class_total = int(class_mask.sum())
                class_valid = int(np.isfinite(column[class_mask]).sum()) if class_total else 0
                per_class[class_label] = {"valid": class_valid, "total": class_total}
            coverage[feature_type] = {
                "valid": valid,
                "total": int(total_rows),
                "valid_fraction": float(valid / total_rows) if total_rows else 0.0,
                "per_class": per_class,
            }
        return coverage
