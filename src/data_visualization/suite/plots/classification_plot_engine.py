"""Matplotlib rendering for classification evaluation artifacts.

Shared by the PDF report generator and the visualization-suite plugin so both
draw identical charts from an EvaluationArtifact.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib.figure import Figure

from src.classification.evaluation.evaluation_artifact import (
    EvaluationArtifact,
    ModelFamilyResult,
)
from src.classification.evaluation.metrics import ranking_score

_BASIS_KEYS = ["stride", "epoch"]


@dataclass
class ClassificationPlotEngine:
    figure: Figure

    def _find_model(
        self, artifact: EvaluationArtifact, model_name: str
    ) -> Optional[ModelFamilyResult]:
        for result in artifact.model_results:
            if result.model_name == model_name:
                return result
        return None

    @staticmethod
    def best_fusion(result: ModelFamilyResult) -> Optional[str]:
        if not result.fusion_results:
            return None
        return max(
            result.fusion_results.items(),
            key=lambda item: ranking_score(item[1]),
        )[0]

    def render_model_comparison(
        self,
        artifact: EvaluationArtifact,
        metric: str = "roc_auc",
        fusion: Optional[str] = None,
    ) -> None:
        """Bar chart of one metric across model families (best fusion per family)."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if not artifact.model_results:
            self._empty(axis, "No model results available")
            return

        names: List[str] = []
        means: List[float] = []
        errors: List[float] = []
        for result in artifact.model_results:
            chosen = fusion or self.best_fusion(result)
            metrics = result.fusion_results.get(chosen, {})
            mean = metrics.get(f"{metric}_mean", np.nan)
            std = metrics.get(f"{metric}_std", 0.0)
            names.append(f"{result.model_name}\n[{chosen}]")
            means.append(mean if np.isfinite(mean) else 0.0)
            errors.append(std if np.isfinite(std) else 0.0)

        positions = range(len(names))
        axis.bar(list(positions), means, yerr=errors, capsize=4, color="#4c72b0")
        axis.set_xticks(list(positions))
        axis.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        axis.set_ylabel(f"{metric} (mean)")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"Model family comparison - {metric}")
        axis.grid(alpha=0.2, axis="y")
        self.figure.tight_layout()

    def render_roc_overlay(self, artifact: EvaluationArtifact, model_name: str) -> None:
        self._curve_overlay(artifact, model_name, "roc")

    def render_pr_overlay(self, artifact: EvaluationArtifact, model_name: str) -> None:
        self._curve_overlay(artifact, model_name, "pr")

    def _curve_overlay(
        self, artifact: EvaluationArtifact, model_name: str, kind: str
    ) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        result = self._find_model(artifact, model_name)
        if result is None:
            self._empty(axis, f"No results for {model_name}")
            return

        curves = result.roc_curves if kind == "roc" else result.pr_curves
        plotted = False
        for fusion_name, curve in curves.items():
            if kind == "roc":
                x = curve.get("fpr", [])
                y = curve.get("tpr", [])
            else:
                x = curve.get("recall", [])
                y = curve.get("precision", [])
            if x and y:
                axis.plot(x, y, label=fusion_name, linewidth=1.5)
                plotted = True

        if not plotted:
            self._empty(axis, "No curve data (single-class folds?)")
            return

        if kind == "roc":
            axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
            axis.set_title(f"ROC curves - {model_name}")
        else:
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.set_title(f"Precision-Recall curves - {model_name}")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=7)
        self.figure.tight_layout()

    def render_fusion_comparison(
        self, artifact: EvaluationArtifact, model_name: str, metric: str = "roc_auc"
    ) -> None:
        """Compare fusion strategies (and base learners) for one family."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        result = self._find_model(artifact, model_name)
        if result is None:
            self._empty(axis, f"No results for {model_name}")
            return

        labels: List[str] = []
        means: List[float] = []
        errors: List[float] = []
        colors: List[str] = []
        for basis_key in _BASIS_KEYS:
            metrics = result.base_results.get(basis_key, {})
            labels.append(f"{basis_key}\n(base)")
            means.append(_safe(metrics.get(f"{metric}_mean")))
            errors.append(_safe(metrics.get(f"{metric}_std")))
            colors.append("#999999")
        for fusion_name, metrics in result.fusion_results.items():
            labels.append(fusion_name)
            means.append(_safe(metrics.get(f"{metric}_mean")))
            errors.append(_safe(metrics.get(f"{metric}_std")))
            colors.append("#4c72b0")

        positions = range(len(labels))
        axis.bar(list(positions), means, yerr=errors, capsize=4, color=colors)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        axis.set_ylabel(f"{metric} (mean)")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"Base vs fusion - {model_name} ({metric})")
        axis.grid(alpha=0.2, axis="y")
        self.figure.tight_layout()

    def render_confusion(
        self, artifact: EvaluationArtifact, model_name: str, fusion: Optional[str] = None
    ) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        result = self._find_model(artifact, model_name)
        if result is None:
            self._empty(axis, f"No results for {model_name}")
            return
        fusion = fusion or self.best_fusion(result)
        counts = result.confusion.get(fusion, {})
        matrix = np.array(
            [
                [_safe(counts.get("tn")), _safe(counts.get("fp"))],
                [_safe(counts.get("fn")), _safe(counts.get("tp"))],
            ]
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Pred non-faller", "Pred faller"], fontsize=8)
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["True non-faller", "True faller"], fontsize=8)
        for row in range(2):
            for col in range(2):
                axis.text(
                    col,
                    row,
                    f"{int(matrix[row, col])}",
                    ha="center",
                    va="center",
                    color="black",
                )
        axis.set_title(f"Confusion (summed) - {model_name} [{fusion}]")
        self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        self.figure.tight_layout()

    def render_ranking_table(self, artifact: EvaluationArtifact, top_n: int = 12) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_axis_off()
        if not artifact.ranking:
            self._empty(axis, "No ranking available")
            return
        rows = artifact.ranking[:top_n]
        table_data = [
            [
                f"{index + 1}",
                row["model_name"],
                row["fusion"],
                _fmt(row.get("roc_auc_mean")),
                _fmt(row.get("pr_auc_mean")),
                _fmt(row.get("balanced_accuracy_mean")),
                _fmt(row.get("score")),
            ]
            for index, row in enumerate(rows)
        ]
        table = axis.table(
            cellText=table_data,
            colLabels=["#", "Model", "Fusion", "ROC-AUC", "PR-AUC", "Bal Acc", "Score"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        axis.set_title("Model + fusion ranking")
        self.figure.tight_layout()

    def _empty(self, axis, message: str) -> None:
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self.figure.tight_layout()


def _safe(value) -> float:
    if value is None:
        return 0.0
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _fmt(value) -> str:
    if value is None:
        return "-"
    value = float(value)
    return "-" if not np.isfinite(value) else f"{value:.3f}"
