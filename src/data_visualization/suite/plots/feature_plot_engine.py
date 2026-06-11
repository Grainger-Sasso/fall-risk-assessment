from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.figure import Figure

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.services.feature_quality_report import (
    FeatureLevelQualityReport,
    ParticipantLevelQualityReport,
)
from src.data_visualization.suite.services.visualization_data_service import (
    PerRecordCoverageMatrix,
)


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

    def render_feature_missingness(
        self,
        report: FeatureLevelQualityReport,
        top_n: Optional[int] = None,
    ) -> Dict[FeatureType, float]:
        """Bar chart of per-feature missing fraction among usable samples."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if report.is_empty:
            ax.text(0.5, 0.5, "No feature values available", ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        ordered = sorted(report.metrics, key=lambda m: m.missing_fraction, reverse=True)
        if top_n is not None and top_n > 0:
            ordered = ordered[:top_n]
        names = [metric.feature_type.value for metric in ordered]
        fractions = [metric.missing_fraction for metric in ordered]
        positions = range(len(ordered))
        colors = [
            "#d62728" if fraction >= 0.5 else ("#ff7f0e" if fraction > 0.0 else "#2ca02c")
            for fraction in fractions
        ]

        ax.barh(list(positions), fractions, color=colors)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Missing fraction (usable samples only)")
        ax.set_title(f"Feature missingness ({report.basis.value})")
        ax.grid(alpha=0.2, axis="x")
        self.figure.tight_layout()
        return {metric.feature_type: metric.missing_fraction for metric in ordered}

    def render_feature_separability(
        self,
        report: FeatureLevelQualityReport,
        top_n: Optional[int] = None,
    ) -> Dict[FeatureType, float]:
        """Cohen's d bar chart from a feature-level quality report."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not report.separation or len(report.classes) != 2:
            ax.text(
                0.5,
                0.5,
                "Class separability requires exactly two populated classes",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        ordered = sorted(report.separation.items(), key=lambda kv: abs(kv[1]), reverse=True)
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
        ax.set_xlabel(f"Cohen's d  ({report.classes[0]} minus {report.classes[1]})")
        ax.set_title(f"Class separability by feature ({report.basis.value})")
        ax.grid(alpha=0.2, axis="x")
        self.figure.tight_layout()
        return dict(ordered)

    def render_feature_correlation_from_report(
        self,
        report: FeatureLevelQualityReport,
    ) -> np.ndarray:
        """Correlation heatmap from a pre-built feature-level report."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        correlation = np.asarray(report.correlation_matrix, dtype=float)

        if correlation.size == 0 or not report.feature_types:
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
        labels = [feature_type.value for feature_type in report.feature_types]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"Feature correlation ({report.basis.value})")
        self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
        self.figure.tight_layout()
        return correlation

    def render_participant_sample_counts(
        self,
        report: ParticipantLevelQualityReport,
        sort_by: str = "count_asc",
    ) -> Dict[str, int]:
        """Per-participant usable sample counts, colored by class."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if report.is_empty:
            ax.text(0.5, 0.5, "No participants available", ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        participants = list(report.participants)
        if sort_by == "count_asc":
            participants.sort(key=lambda m: (m.usable_sample_count, m.participant_id))
        elif sort_by == "count_desc":
            participants.sort(
                key=lambda m: (-m.usable_sample_count, m.participant_id)
            )
        elif sort_by == "participant":
            participants.sort(key=lambda m: m.participant_id.lower())
        elif sort_by == "coverage":
            participants.sort(key=lambda m: (m.mean_coverage, m.participant_id))
        else:
            raise ValueError(f"Unsupported sort_by '{sort_by}'")

        labels = [
            f"{metric.participant_id} ({metric.usable_sample_count})"
            for metric in participants
        ]
        counts = [metric.usable_sample_count for metric in participants]
        class_colors = {
            "faller": "#d62728",
            "non-faller": "#2ca02c",
        }
        colors = [
            class_colors.get(metric.class_label, "#4c72b0") for metric in participants
        ]

        ax.barh(range(len(participants)), counts, color=colors)
        ax.set_yticks(range(len(participants)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Usable samples (padding excluded)")
        ax.set_title(f"Per-participant sample counts ({report.basis.value})")
        ax.grid(alpha=0.2, axis="x")
        self.figure.tight_layout()
        return {metric.participant_id: metric.usable_sample_count for metric in participants}

    def render_participant_coverage_heatmap(
        self,
        report: ParticipantLevelQualityReport,
        sort_by: str = "worst_first",
    ) -> Dict[str, Union[int, float, List[str]]]:
        """Heatmap of per-participant, per-feature finite-value fractions."""
        self.figure.clear()

        if report.is_empty:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No participants available", ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        matrix, row_labels, order = self._order_participant_coverage(report, sort_by)
        feature_labels = [feature_type.value for feature_type in report.feature_types]

        ax_heat = self.figure.add_subplot(111)
        image = ax_heat.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlGn",
            interpolation="nearest",
        )
        ax_heat.set_xticks(range(len(feature_labels)))
        ax_heat.set_xticklabels(feature_labels, rotation=90, fontsize=6)
        ax_heat.set_yticks(range(len(row_labels)))
        ax_heat.set_yticklabels(row_labels, fontsize=7)
        ax_heat.set_xlabel("Feature type")
        ax_heat.set_ylabel("Participant (usable samples)")
        ax_heat.set_title(
            f"Per-participant feature coverage ({report.basis.value})"
        )
        colorbar = self.figure.colorbar(image, ax=ax_heat, fraction=0.02, pad=0.02)
        colorbar.set_label("Finite value fraction")

        self.figure.tight_layout()
        return self._summarize_participant_coverage(report, order)

    @staticmethod
    def _order_participant_coverage(
        report: ParticipantLevelQualityReport,
        sort_by: str,
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        indices = list(range(len(report.participants)))
        if sort_by == "worst_first":
            indices.sort(
                key=lambda idx: (
                    report.participants[idx].mean_coverage,
                    report.participants[idx].participant_id,
                )
            )
        elif sort_by == "participant":
            indices.sort(
                key=lambda idx: report.participants[idx].participant_id.lower()
            )
        elif sort_by == "count_asc":
            indices.sort(
                key=lambda idx: (
                    report.participants[idx].usable_sample_count,
                    report.participants[idx].participant_id,
                )
            )
        elif sort_by == "count_desc":
            indices.sort(
                key=lambda idx: (
                    -report.participants[idx].usable_sample_count,
                    report.participants[idx].participant_id,
                )
            )
        else:
            raise ValueError(f"Unsupported sort_by '{sort_by}'")

        matrix = np.vstack(
            [report.participants[idx].per_feature_fractions for idx in indices]
        )
        row_labels = [
            (
                f"{report.participants[idx].participant_id} "
                f"(n={report.participants[idx].usable_sample_count})"
            )
            for idx in indices
        ]
        return matrix, row_labels, indices

    @staticmethod
    def _summarize_participant_coverage(
        report: ParticipantLevelQualityReport,
        order: List[int],
    ) -> Dict[str, Union[int, float, List[str]]]:
        mean_per_participant = [
            report.participants[idx].mean_coverage for idx in order
        ]
        participants_with_gaps = int(
            sum(1 for value in mean_per_participant if value < 1.0)
        )
        worst_indices = sorted(
            order,
            key=lambda idx: (
                report.participants[idx].mean_coverage,
                report.participants[idx].participant_id,
            ),
        )[:5]
        worst_participants = [
            (
                f"{report.participants[idx].participant_id} "
                f"(mean={report.participants[idx].mean_coverage:.2f}, "
                f"usable={report.participants[idx].usable_sample_count})"
            )
            for idx in worst_indices
            if report.participants[idx].mean_coverage < 1.0
        ]
        total_usable = sum(metric.usable_sample_count for metric in report.participants)
        total_padded = sum(metric.padded_slot_count for metric in report.participants)
        return {
            "n_participants": len(report.participants),
            "n_feature_types": len(report.feature_types),
            "total_usable_samples": total_usable,
            "total_padded_slots": total_padded,
            "participants_with_any_missing": participants_with_gaps,
            "inconsistent_count_records": len(report.inconsistent_count_records),
            "worst_participants": worst_participants,
        }

    def render_feature_coverage(
        self,
        feature_matrix: np.ndarray,
        row_class_labels: List[str],
        feature_types: List[FeatureType],
        basis: SampleBasis,
    ) -> Dict[FeatureType, Dict[str, float]]:
        """Render the fraction of finite values per feature among usable samples."""
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
        ax.set_xlabel("Finite value fraction (usable samples only)")
        ax.set_title(f"Feature coverage among populated samples ({basis.value})")
        ax.grid(alpha=0.2, axis="x")
        self.figure.tight_layout()
        return coverage

    def render_per_record_missingness_heatmap(
        self,
        coverage: PerRecordCoverageMatrix,
        basis: SampleBasis,
        sort_by: str = "worst_first",
        row_label_mode: str = "participant",
    ) -> Dict[str, Union[int, float, List[str]]]:
        """
        Heatmap of valid-sample fraction per feature file (row) and feature type
        (column). Helps spot which ``features_*.h5`` files carry missing data.
        """
        self.figure.clear()

        if coverage.is_empty:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No feature records available", ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            return {}

        matrix, row_labels, order = self._order_per_record_coverage(
            coverage, sort_by=sort_by, row_label_mode=row_label_mode
        )
        feature_labels = [feature_type.value for feature_type in coverage.feature_types]

        ax_heat = self.figure.add_subplot(111)
        image = ax_heat.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlGn",
            interpolation="nearest",
        )
        ax_heat.set_xticks(range(len(feature_labels)))
        ax_heat.set_xticklabels(feature_labels, rotation=90, fontsize=6)
        ax_heat.set_yticks(range(len(row_labels)))
        ax_heat.set_yticklabels(row_labels, fontsize=7)
        ax_heat.set_xlabel("Feature type")
        ax_heat.set_ylabel("Feature record")
        ax_heat.set_title(
            f"Per-record feature coverage among populated samples ({basis.value})"
        )
        colorbar = self.figure.colorbar(image, ax=ax_heat, fraction=0.02, pad=0.02)
        colorbar.set_label("Finite value fraction")

        self.figure.tight_layout()
        return self._summarize_per_record_coverage(coverage, order)

    @staticmethod
    def _order_per_record_coverage(
        coverage: PerRecordCoverageMatrix,
        sort_by: str,
        row_label_mode: str,
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        indices = list(range(coverage.n_records))
        if sort_by == "worst_first":
            mean_coverage = np.nanmean(coverage.matrix, axis=1)
            indices.sort(key=lambda idx: (mean_coverage[idx], coverage.participant_ids[idx]))
        elif sort_by == "participant":
            indices.sort(key=lambda idx: coverage.participant_ids[idx].lower())
        elif sort_by == "feature_id":
            indices.sort(key=lambda idx: coverage.feature_ids[idx].lower())
        else:
            raise ValueError(f"Unsupported sort_by '{sort_by}'")

        matrix = coverage.matrix[indices, :]
        if row_label_mode == "participant":
            row_labels = [coverage.participant_ids[idx] for idx in indices]
        elif row_label_mode == "feature_id":
            row_labels = [coverage.feature_ids[idx] for idx in indices]
        else:
            raise ValueError(f"Unsupported row_label_mode '{row_label_mode}'")
        return matrix, row_labels, indices

    @staticmethod
    def _summarize_per_record_coverage(
        coverage: PerRecordCoverageMatrix,
        order: List[int],
    ) -> Dict[str, Union[int, float, List[str]]]:
        mean_per_record = np.nanmean(coverage.matrix, axis=1)
        records_with_gaps = int(np.sum(mean_per_record < 1.0))
        records_with_empty_features = int(
            np.sum(np.any(coverage.matrix == 0.0, axis=1))
        )
        worst_indices = sorted(
            order,
            key=lambda idx: (mean_per_record[idx], coverage.participant_ids[idx]),
        )[:5]
        worst_records = [
            (
                f"{coverage.participant_ids[idx]} "
                f"(mean={mean_per_record[idx]:.2f}, id={coverage.feature_ids[idx]})"
            )
            for idx in worst_indices
            if mean_per_record[idx] < 1.0
        ]
        total_usable = int(np.sum(coverage.usable_sample_counts)) if coverage.usable_sample_counts else 0
        return {
            "n_records": coverage.n_records,
            "n_feature_types": len(coverage.feature_types),
            "total_usable_samples": total_usable,
            "records_with_any_missing": records_with_gaps,
            "records_with_fully_missing_feature": records_with_empty_features,
            "worst_records": worst_records,
        }

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
        """Per-feature finite-value counts and fractions among usable sample rows."""
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
