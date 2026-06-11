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
from src.data_visualization.suite.services.feature_quality_report import (
    FeatureLevelQualityReport,
    ParticipantLevelQualityReport,
    feature_level_summary_text,
    participant_level_summary_text,
)

LEVEL_FEATURE = "Feature level"
LEVEL_PARTICIPANT = "Participant level"

VIEW_FEATURE_MISSINGNESS = "Missingness"
VIEW_FEATURE_CORRELATION = "Correlation"
VIEW_FEATURE_SEPARABILITY = "Class separability"

VIEW_PARTICIPANT_COUNTS = "Usable sample counts"
VIEW_PARTICIPANT_COVERAGE = "Per-feature coverage"

SORT_WORST_FIRST = "Worst coverage first"
SORT_PARTICIPANT = "Participant ID"
SORT_COUNT_ASC = "Fewest samples first"
SORT_COUNT_DESC = "Most samples first"


class FeatureQualityPlugin(VisualizationPlugin):
    """Two-level feature quality reports: population features and participants."""

    plugin_id = "feature_quality"
    display_name = "Feature Quality Report"

    def __init__(self) -> None:
        self._context: Optional[PluginContext] = None
        self._feature_report: Optional[FeatureLevelQualityReport] = None
        self._participant_report: Optional[ParticipantLevelQualityReport] = None
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
        controls.setMinimumWidth(360)
        layout = QVBoxLayout(controls)

        self.level_selector = QComboBox(controls)
        self.level_selector.addItems([LEVEL_FEATURE, LEVEL_PARTICIPANT])
        self.level_selector.currentIndexChanged.connect(self._on_level_changed)

        self.basis_selector = QComboBox(controls)
        self.basis_selector.addItems([SampleBasis.EPOCH.value, SampleBasis.STRIDE.value])

        self.view_selector = QComboBox(controls)
        self.view_selector.currentIndexChanged.connect(self._on_view_changed)
        self._populate_view_selector()

        self.sort_selector = QComboBox(controls)
        self._populate_sort_selector()

        render_button = QPushButton("Render report", controls)
        render_button.clicked.connect(self._render_current)

        form = QFormLayout()
        form.addRow("Report level", self.level_selector)
        form.addRow("Sample basis", self.basis_selector)
        form.addRow("View", self.view_selector)
        form.addRow("Sort", self.sort_selector)

        self.summary_label = QLabel(
            "Select a report level and click Render to build the quality report.",
            controls,
        )
        self.summary_label.setWordWrap(True)

        layout.addWidget(QLabel("Feature quality report", controls))
        layout.addLayout(form)
        layout.addWidget(render_button)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Report summary", controls))
        layout.addWidget(self.summary_label)
        layout.addStretch(1)
        return controls

    def _on_level_changed(self) -> None:
        self._populate_view_selector()
        self._populate_sort_selector()

    def _on_view_changed(self) -> None:
        if self.level_selector.currentText() == LEVEL_PARTICIPANT:
            self._populate_sort_selector()

    def _populate_view_selector(self) -> None:
        self.view_selector.blockSignals(True)
        self.view_selector.clear()
        if self.level_selector.currentText() == LEVEL_FEATURE:
            self.view_selector.addItems(
                [
                    VIEW_FEATURE_MISSINGNESS,
                    VIEW_FEATURE_CORRELATION,
                    VIEW_FEATURE_SEPARABILITY,
                ]
            )
        else:
            self.view_selector.addItems(
                [
                    VIEW_PARTICIPANT_COUNTS,
                    VIEW_PARTICIPANT_COVERAGE,
                ]
            )
        self.view_selector.blockSignals(False)

    def _populate_sort_selector(self) -> None:
        self.sort_selector.blockSignals(True)
        self.sort_selector.clear()
        if self.level_selector.currentText() == LEVEL_FEATURE:
            self.sort_selector.addItem("N/A")
            self.sort_selector.setEnabled(False)
        else:
            self.sort_selector.setEnabled(True)
            view = self.view_selector.currentText()
            if view == VIEW_PARTICIPANT_COUNTS:
                self.sort_selector.addItems(
                    [SORT_COUNT_ASC, SORT_COUNT_DESC, SORT_PARTICIPANT]
                )
            else:
                self.sort_selector.addItems(
                    [SORT_WORST_FIRST, SORT_PARTICIPANT, SORT_COUNT_ASC, SORT_COUNT_DESC]
                )
        self.sort_selector.blockSignals(False)

    def _render_current(self) -> None:
        if self._context is None:
            return

        basis = SampleBasis(self.basis_selector.currentText())
        if self.level_selector.currentText() == LEVEL_FEATURE:
            self._feature_report = self._context.data_service.build_feature_level_quality_report(
                basis
            )
            self._render_feature_level(self._feature_report)
        else:
            self._participant_report = (
                self._context.data_service.build_participant_level_quality_report(basis)
            )
            self._render_participant_level(self._participant_report)

    def _render_feature_level(self, report: FeatureLevelQualityReport) -> None:
        self.summary_label.setText(feature_level_summary_text(report))
        view = self.view_selector.currentText()

        if report.is_empty:
            self._render_placeholder("No usable samples for this basis")
            return

        if view == VIEW_FEATURE_MISSINGNESS:
            self._plot_engine.render_feature_missingness(report)
        elif view == VIEW_FEATURE_CORRELATION:
            self._plot_engine.render_feature_correlation_from_report(report)
        elif view == VIEW_FEATURE_SEPARABILITY:
            self._plot_engine.render_feature_separability(report)
        else:
            self._render_placeholder(f"Unsupported view: {view}")
            return

        self._canvas.draw_idle()

    def _render_participant_level(self, report: ParticipantLevelQualityReport) -> None:
        self.summary_label.setText(participant_level_summary_text(report))
        view = self.view_selector.currentText()

        if report.is_empty:
            self._render_placeholder("No feature records for this basis")
            return

        sort_key = self._participant_sort_key()
        if view == VIEW_PARTICIPANT_COUNTS:
            self._plot_engine.render_participant_sample_counts(report, sort_by=sort_key)
        elif view == VIEW_PARTICIPANT_COVERAGE:
            summary = self._plot_engine.render_participant_coverage_heatmap(
                report, sort_by=sort_key
            )
            self.summary_label.setText(
                participant_level_summary_text(report)
                + "\n\n"
                + self._format_participant_plot_summary(summary)
            )
        else:
            self._render_placeholder(f"Unsupported view: {view}")
            return

        self._canvas.draw_idle()

    def _participant_sort_key(self) -> str:
        mapping = {
            SORT_WORST_FIRST: "worst_first",
            SORT_PARTICIPANT: "participant",
            SORT_COUNT_ASC: "count_asc",
            SORT_COUNT_DESC: "count_desc",
        }
        return mapping[self.sort_selector.currentText()]

    def _render_placeholder(self, message: str = "Select options and click Render") -> None:
        self._figure.clear()
        axis = self._figure.add_subplot(111)
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self._figure.tight_layout()
        self._canvas.draw_idle()

    @staticmethod
    def _format_participant_plot_summary(
        summary: Dict[str, Union[int, float, List[str]]],
    ) -> str:
        if not summary:
            return ""
        lines = [
            f"Participants plotted: {int(summary['n_participants'])}",
            f"Total usable samples: {int(summary['total_usable_samples'])}",
            f"Padded tensor slots excluded: {int(summary['total_padded_slots'])}",
            (
                "Participants with missing feature values: "
                f"{int(summary['participants_with_any_missing'])}"
            ),
        ]
        if int(summary.get("inconsistent_count_records", 0)):
            lines.append(
                "Count mismatches detected: "
                f"{int(summary['inconsistent_count_records'])}"
            )
        worst = summary.get("worst_participants", [])
        if worst:
            lines.append("Lowest coverage:")
            lines.extend(f"  - {entry}" for entry in worst)
        return "\n".join(lines)
