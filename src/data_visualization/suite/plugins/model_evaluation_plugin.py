from pathlib import Path
from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.classification.evaluation.evaluation_artifact import EvaluationArtifact
from src.data_visualization.suite.plots.classification_plot_engine import (
    ClassificationPlotEngine,
)
from src.data_visualization.suite.plugins.base import PluginContext, VisualizationPlugin

VIEW_MODEL_COMPARISON_ROC = "Model Comparison (ROC-AUC)"
VIEW_MODEL_COMPARISON_PR = "Model Comparison (PR-AUC)"
VIEW_RANKING = "Ranking Table"
VIEW_FUSION_COMPARISON = "Base vs Fusion (selected model)"
VIEW_ROC = "ROC Curves (selected model)"
VIEW_PR = "PR Curves (selected model)"
VIEW_CONFUSION = "Confusion Matrix (selected model)"


class ModelEvaluationPlugin(VisualizationPlugin):
    """Load and visualize classification evaluation artifacts (JSON)."""

    plugin_id = "model_evaluation"
    display_name = "Model Evaluation"

    def __init__(self) -> None:
        self._context: Optional[PluginContext] = None
        self._artifact: Optional[EvaluationArtifact] = None
        self._figure = Figure(figsize=(10, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._plot_engine = ClassificationPlotEngine(self._figure)

    def build_panel(self, parent: QWidget, context: PluginContext) -> QWidget:
        self._context = context
        panel = QWidget(parent)
        root_layout = QHBoxLayout(panel)
        root_layout.addWidget(self._build_controls(panel), stretch=0)
        root_layout.addWidget(self._canvas, stretch=1)
        return panel

    def _build_controls(self, parent: QWidget) -> QWidget:
        controls = QWidget(parent)
        controls.setMinimumWidth(360)
        layout = QVBoxLayout(controls)

        self.path_box = QLineEdit(controls)
        self.path_box.setPlaceholderText("Path to classification artifact JSON")

        browse_button = QPushButton("Browse...", controls)
        browse_button.clicked.connect(self._browse)
        load_button = QPushButton("Load Artifact", controls)
        load_button.clicked.connect(self._load_artifact)

        self.view_selector = QComboBox(controls)
        self.view_selector.addItems(
            [
                VIEW_MODEL_COMPARISON_ROC,
                VIEW_MODEL_COMPARISON_PR,
                VIEW_RANKING,
                VIEW_FUSION_COMPARISON,
                VIEW_ROC,
                VIEW_PR,
                VIEW_CONFUSION,
            ]
        )
        self.model_selector = QComboBox(controls)

        render_button = QPushButton("Render", controls)
        render_button.clicked.connect(self._render_current)

        form = QFormLayout()
        form.addRow("View", self.view_selector)
        form.addRow("Model", self.model_selector)

        self.status_label = QLabel("No artifact loaded.", controls)
        self.status_label.setWordWrap(True)

        layout.addWidget(QLabel("Evaluation artifact", controls))
        layout.addWidget(self.path_box)
        layout.addWidget(browse_button)
        layout.addWidget(load_button)
        layout.addSpacing(12)
        layout.addLayout(form)
        layout.addWidget(render_button)
        layout.addSpacing(12)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        return controls

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._canvas, "Select evaluation artifact", "", "JSON files (*.json)"
        )
        if path:
            self.path_box.setText(path)

    def _load_artifact(self) -> None:
        path_text = self.path_box.text().strip()
        if not path_text:
            self.status_label.setText("Provide an artifact JSON path first.")
            return
        path = Path(path_text)
        if not path.exists():
            self.status_label.setText(f"File not found: {path}")
            return
        try:
            self._artifact = EvaluationArtifact.from_json(path)
        except Exception as exc:  # pragma: no cover - UI safety
            self.status_label.setText(f"Failed to load artifact: {exc}")
            return

        self.model_selector.clear()
        self.model_selector.addItems(
            [result.model_name for result in self._artifact.model_results]
        )
        counts = self._artifact.participant_counts
        self.status_label.setText(
            f"Loaded {len(self._artifact.model_results)} model families. "
            f"Participants: {counts.get('common', 0)} "
            f"(faller={counts.get('faller', 0)}, non-faller={counts.get('non_faller', 0)})."
        )
        self._render_current()

    def _render_current(self) -> None:
        if self._artifact is None:
            self.status_label.setText("Load an artifact first.")
            return

        view = self.view_selector.currentText()
        model_name = self.model_selector.currentText().strip()

        if view == VIEW_MODEL_COMPARISON_ROC:
            self._plot_engine.render_model_comparison(self._artifact, metric="roc_auc")
        elif view == VIEW_MODEL_COMPARISON_PR:
            self._plot_engine.render_model_comparison(self._artifact, metric="pr_auc")
        elif view == VIEW_RANKING:
            self._plot_engine.render_ranking_table(self._artifact)
        elif not model_name:
            self.status_label.setText("Select a model for this view.")
            return
        elif view == VIEW_FUSION_COMPARISON:
            self._plot_engine.render_fusion_comparison(self._artifact, model_name)
        elif view == VIEW_ROC:
            self._plot_engine.render_roc_overlay(self._artifact, model_name)
        elif view == VIEW_PR:
            self._plot_engine.render_pr_overlay(self._artifact, model_name)
        elif view == VIEW_CONFUSION:
            self._plot_engine.render_confusion(self._artifact, model_name)

        self._canvas.draw_idle()
