from typing import Dict, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.plots.feature_plot_engine import FeaturePlotEngine
from src.data_visualization.suite.plugins.base import PluginContext, VisualizationPlugin


VIEW_CLASS_VIOLIN = "Class Violin (selected feature)"


class FeatureAnalyticsPlugin(VisualizationPlugin):
    """Feature distribution analytics with class-based and population views."""

    plugin_id = "feature_analytics"
    display_name = "Feature Analytics"

    def __init__(self) -> None:
        self._context: Optional[PluginContext] = None
        self._feature_ids = []
        self._selected_feature_type: Optional[FeatureType] = None
        self._figure = Figure(figsize=(10, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._plot_engine = FeaturePlotEngine(self._figure)

    def build_panel(self, parent: QWidget, context: PluginContext) -> QWidget:
        self._context = context
        panel = QWidget(parent)
        root_layout = QHBoxLayout(panel)
        controls = self._build_controls(panel)
        root_layout.addWidget(controls, stretch=0)
        root_layout.addWidget(self._canvas, stretch=1)
        self._load_feature_ids()
        return panel

    def _build_controls(self, parent: QWidget) -> QWidget:
        controls = QWidget(parent)
        controls.setMinimumWidth(340)
        layout = QVBoxLayout(controls)

        self.search_box = QLineEdit(controls)
        self.search_box.setPlaceholderText("Filter feature IDs")
        self.search_box.textChanged.connect(self._filter_feature_ids)

        self.feature_selector = QComboBox(controls)
        self.feature_selector.currentIndexChanged.connect(self._on_basis_or_feature_changed)

        self.basis_selector = QComboBox(controls)
        self.basis_selector.addItems([SampleBasis.EPOCH.value, SampleBasis.STRIDE.value])
        self.basis_selector.currentIndexChanged.connect(self._on_basis_or_feature_changed)

        self.view_selector = QComboBox(controls)
        self.view_selector.addItems([VIEW_CLASS_VIOLIN])

        self.feature_type_selector = QComboBox(controls)
        self.feature_type_selector.currentIndexChanged.connect(self._on_feature_type_selected)

        render_button = QPushButton("Render", controls)
        render_button.clicked.connect(self._render_current)

        form = QFormLayout()
        form.addRow("View", self.view_selector)
        form.addRow("Feature record", self.feature_selector)
        form.addRow("Basis", self.basis_selector)
        form.addRow("Feature type", self.feature_type_selector)

        self.summary_label = QLabel("-", controls)
        self.summary_label.setWordWrap(True)

        layout.addWidget(QLabel("Feature visualization controls", controls))
        layout.addWidget(self.search_box)
        layout.addLayout(form)
        layout.addWidget(render_button)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Summary statistics", controls))
        layout.addWidget(self.summary_label)
        layout.addStretch(1)
        return controls

    def _load_feature_ids(self) -> None:
        if self._context is None:
            return
        self._feature_ids = sorted(
            self._context.data_service.list_feature_ids(), key=lambda item: item.value
        )
        self.feature_selector.clear()
        self.feature_selector.addItems([item.value for item in self._feature_ids])
        self._refresh_feature_types()

    def _filter_feature_ids(self, text: str) -> None:
        filtered = [
            item for item in self._feature_ids if text.lower().strip() in item.value.lower()
        ]
        self.feature_selector.blockSignals(True)
        self.feature_selector.clear()
        self.feature_selector.addItems([item.value for item in filtered])
        self.feature_selector.blockSignals(False)
        self._refresh_feature_types()

    def _on_basis_or_feature_changed(self) -> None:
        self._refresh_feature_types()

    def _refresh_feature_types(self) -> None:
        if self._context is None:
            return
        selected_feature_id = self.feature_selector.currentText().strip()
        if not selected_feature_id:
            self.feature_type_selector.clear()
            self._selected_feature_type = None
            return
        basis = SampleBasis(self.basis_selector.currentText())
        selected_feature_identifier = next(
            (item for item in self._feature_ids if item.value == selected_feature_id),
            None,
        )
        if selected_feature_identifier is None:
            self.feature_type_selector.clear()
            self._selected_feature_type = None
            return
        feature_types = self._context.data_service.list_feature_types_for_basis(
            feature_id=selected_feature_identifier,
            basis=basis,
        )
        self.feature_type_selector.blockSignals(True)
        self.feature_type_selector.clear()
        self.feature_type_selector.addItems([item.value for item in feature_types])
        self.feature_type_selector.blockSignals(False)
        if feature_types:
            self._selected_feature_type = feature_types[0]
        else:
            self._selected_feature_type = None

    def _on_feature_type_selected(self, index: int) -> None:
        if index < 0:
            self._selected_feature_type = None
            return
        selected_value = self.feature_type_selector.currentText()
        try:
            self._selected_feature_type = FeatureType(selected_value)
        except ValueError:
            self._selected_feature_type = None

    def _render_current(self) -> None:
        if self._context is None:
            return

        basis = SampleBasis(self.basis_selector.currentText())
        self._render_class_violin(basis)

    def _render_class_violin(self, basis: SampleBasis) -> None:
        if self._selected_feature_type is None:
            self.summary_label.setText("No feature type selected.")
            self._render_empty("No feature type selected")
            return
        grouped_values = self._context.data_service.collect_feature_values_by_class(
            basis=basis,
            feature_type=self._selected_feature_type,
        )
        summary = self._plot_engine.render_class_violin(
            grouped_values=grouped_values,
            feature_type=self._selected_feature_type,
            basis=basis,
        )
        self._canvas.draw_idle()
        self.summary_label.setText(self._format_summary(summary))

    @staticmethod
    def _format_summary(summary: Dict[str, Dict[str, float]]) -> str:
        if not summary:
            return "No values available for this feature/basis."
        lines = []
        for class_label in sorted(summary.keys()):
            stats = summary[class_label]
            lines.append(
                f"{class_label}: n={int(stats['count'])}, mean={stats['mean']:.3f}, "
                f"median={stats['median']:.3f}, std={stats['std']:.3f}, iqr={stats['iqr']:.3f}"
            )
        return "\n".join(lines)
