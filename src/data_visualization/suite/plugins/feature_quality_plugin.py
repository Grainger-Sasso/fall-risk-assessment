from typing import Dict, List, Optional, Union

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.plots.feature_plot_engine import FeaturePlotEngine
from src.data_visualization.suite.plugins.base import PluginContext, VisualizationPlugin

SORT_WORST_FIRST = "Worst coverage first"
SORT_PARTICIPANT = "Participant ID"
SORT_FEATURE_ID = "Feature record ID"

LABEL_PARTICIPANT = "Participant ID"
LABEL_FEATURE_ID = "Feature record ID"


class FeatureQualityPlugin(VisualizationPlugin):
    """Per-record feature missingness and coverage diagnostics."""

    plugin_id = "feature_quality"
    display_name = "Feature Quality"

    def __init__(self) -> None:
        self._context: Optional[PluginContext] = None
        self._figure = Figure(figsize=(12, 8))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._plot_engine = FeaturePlotEngine(self._figure)

    def build_panel(self, parent: QWidget, context: PluginContext) -> QWidget:
        self._context = context
        panel = QWidget(parent)
        root_layout = QHBoxLayout(panel)
        root_layout.addWidget(self._build_controls(panel), stretch=0)
        root_layout.addWidget(self._canvas, stretch=1)
        self._render_placeholder()
        return panel

    def _build_controls(self, parent: QWidget) -> QWidget:
        controls = QWidget(parent)
        controls.setMinimumWidth(340)
        layout = QVBoxLayout(controls)

        self.basis_selector = QComboBox(controls)
        self.basis_selector.addItems([SampleBasis.EPOCH.value, SampleBasis.STRIDE.value])

        self.sort_selector = QComboBox(controls)
        self.sort_selector.addItems(
            [SORT_WORST_FIRST, SORT_PARTICIPANT, SORT_FEATURE_ID]
        )

        self.label_selector = QComboBox(controls)
        self.label_selector.addItems([LABEL_PARTICIPANT, LABEL_FEATURE_ID])

        render_button = QPushButton("Render", controls)
        render_button.clicked.connect(self._render_current)

        form = QFormLayout()
        form.addRow("Sample basis", self.basis_selector)
        form.addRow("Sort rows by", self.sort_selector)
        form.addRow("Row labels", self.label_selector)

        self.summary_label = QLabel(
            "Click Render to scan all feature files for missing values.",
            controls,
        )
        self.summary_label.setWordWrap(True)

        layout.addWidget(QLabel("Feature file missingness", controls))
        layout.addLayout(form)
        layout.addWidget(render_button)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Summary", controls))
        layout.addWidget(self.summary_label)
        layout.addStretch(1)
        return controls

    def _render_current(self) -> None:
        if self._context is None:
            return

        basis = SampleBasis(self.basis_selector.currentText())
        coverage = self._context.data_service.collect_per_record_coverage(basis)
        if coverage.is_empty:
            self.summary_label.setText("No feature records found in the index.")
            self._render_placeholder("No feature records available")
            return

        summary = self._plot_engine.render_per_record_missingness_heatmap(
            coverage=coverage,
            basis=basis,
            sort_by=self._sort_key(),
            row_label_mode=self._label_key(),
        )
        self._canvas.draw_idle()
        self.summary_label.setText(self._format_summary(summary))

    def _sort_key(self) -> str:
        mapping = {
            SORT_WORST_FIRST: "worst_first",
            SORT_PARTICIPANT: "participant",
            SORT_FEATURE_ID: "feature_id",
        }
        return mapping[self.sort_selector.currentText()]

    def _label_key(self) -> str:
        mapping = {
            LABEL_PARTICIPANT: "participant",
            LABEL_FEATURE_ID: "feature_id",
        }
        return mapping[self.label_selector.currentText()]

    def _render_placeholder(self, message: str = "Select options and click Render") -> None:
        self._figure.clear()
        axis = self._figure.add_subplot(111)
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self._figure.tight_layout()
        self._canvas.draw_idle()

    @staticmethod
    def _format_summary(summary: Dict[str, Union[int, float, List[str]]]) -> str:
        if not summary:
            return "No summary available."
        lines = [
            f"Records scanned: {int(summary['n_records'])}",
            f"Feature types: {int(summary['n_feature_types'])}",
            f"Usable samples (padding excluded): {int(summary.get('total_usable_samples', 0))}",
            f"Records with any missing values among usable samples: {int(summary['records_with_any_missing'])}",
            (
                "Records with at least one fully missing feature: "
                f"{int(summary['records_with_fully_missing_feature'])}"
            ),
        ]
        worst_records = summary.get("worst_records", [])
        if worst_records:
            lines.append("Lowest coverage records:")
            lines.extend(f"  - {entry}" for entry in worst_records)
        else:
            lines.append("All records have full per-feature coverage.")
        return "\n".join(lines)
