from dataclasses import dataclass
from typing import Dict

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
