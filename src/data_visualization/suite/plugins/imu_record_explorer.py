from typing import Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.data_model.data.imu.sensor_data import SensorData
from src.data_visualization.suite.plots.imu_plot_engine import IMUPlotEngine
from src.data_visualization.suite.plugins.base import PluginContext, VisualizationPlugin
from src.data_visualization.suite.services.visualization_data_service import IMURecordEntry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier

NO_USER_LABEL = "(no user)"


class IMURecordExplorerPlugin(VisualizationPlugin):
    plugin_id = "imu_record_explorer"
    display_name = "IMU Record Explorer"

    def __init__(self) -> None:
        self._context: Optional[PluginContext] = None
        self._imu_entries: List[IMURecordEntry] = []
        self._loaded_sensor_data: Optional[SensorData] = None
        self._loaded_imu_id: Optional[IMUDataIdentifier] = None
        self._figure = Figure(figsize=(10, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._plot_engine = IMUPlotEngine(self._figure)

    def build_panel(self, parent: QWidget, context: PluginContext) -> QWidget:
        self._context = context
        root = QWidget(parent)
        root_layout = QHBoxLayout(root)
        controls = self._build_controls_panel(root)
        root_layout.addWidget(controls, stretch=0)
        root_layout.addWidget(self._canvas, stretch=1)
        self._load_imu_ids()
        return root

    def _build_controls_panel(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        panel.setMinimumWidth(320)
        layout = QVBoxLayout(panel)

        self.search_box = QLineEdit(panel)
        self.search_box.setPlaceholderText("Filter by user or IMU ID")
        self.search_box.textChanged.connect(self._filter_imu_ids)

        self.imu_selector = QComboBox(panel)
        self.imu_selector.currentIndexChanged.connect(self._on_imu_selection_changed)

        axis_row = QHBoxLayout()
        self.axis_vertical = QCheckBox("Vertical", panel)
        self.axis_vertical.setChecked(True)
        self.axis_mediolateral = QCheckBox("Mediolateral", panel)
        self.axis_mediolateral.setChecked(True)
        self.axis_anteroposterior = QCheckBox("Anteroposterior", panel)
        self.axis_anteroposterior.setChecked(True)
        for checkbox in (
            self.axis_vertical,
            self.axis_mediolateral,
            self.axis_anteroposterior,
        ):
            checkbox.stateChanged.connect(self._render_current)
            axis_row.addWidget(checkbox)

        self.start_spin = QSpinBox(panel)
        self.start_spin.setMinimum(0)
        self.start_spin.valueChanged.connect(self._on_window_changed)
        self.end_spin = QSpinBox(panel)
        self.end_spin.setMinimum(1)
        self.end_spin.valueChanged.connect(self._on_window_changed)

        self.load_render_button = QPushButton("Load & Render", panel)
        self.load_render_button.clicked.connect(self._load_and_render)

        apply_button = QPushButton("Apply Window", panel)
        apply_button.clicked.connect(self._render_current)

        form = QFormLayout()
        form.addRow("IMU record", self.imu_selector)
        form.addRow("Start index", self.start_spin)
        form.addRow("End index", self.end_spin)

        self.meta_imu = QLabel("-", panel)
        self.meta_user = QLabel("-", panel)
        self.meta_instrument = QLabel("-", panel)
        self.meta_spec = QLabel("-", panel)
        self.meta_imu.setWordWrap(True)
        self.meta_user.setWordWrap(True)
        self.meta_instrument.setWordWrap(True)
        self.meta_spec.setWordWrap(True)

        meta_form = QFormLayout()
        meta_form.addRow("IMU ID", self.meta_imu)
        meta_form.addRow("User ID", self.meta_user)
        meta_form.addRow("Instrument", self.meta_instrument)
        meta_form.addRow("Instrument spec", self.meta_spec)

        layout.addWidget(QLabel("Record selection", panel))
        layout.addWidget(self.search_box)
        layout.addLayout(form)
        layout.addLayout(axis_row)
        layout.addWidget(self.load_render_button)
        layout.addWidget(apply_button)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Metadata", panel))
        layout.addLayout(meta_form)
        layout.addStretch(1)
        return panel

    def _load_imu_ids(self) -> None:
        if self._context is None:
            return
        self._imu_entries = self._context.data_service.list_imu_records_with_users()
        self._populate_selector(self._imu_entries)
        self._loaded_sensor_data = None
        self._loaded_imu_id = None
        self._clear_metadata()
        self._render_empty("Select an IMU record and click Load & Render")

    def _populate_selector(self, entries: List[IMURecordEntry]) -> None:
        self.imu_selector.blockSignals(True)
        self.imu_selector.clear()
        for entry in entries:
            self.imu_selector.addItem(self._entry_label(entry), entry.imu_id.value)
        self.imu_selector.blockSignals(False)

    @staticmethod
    def _entry_label(entry: IMURecordEntry) -> str:
        user_label = entry.user_id.value if entry.user_id is not None else NO_USER_LABEL
        return f"{user_label}  —  {entry.imu_id.value}"

    def _selected_imu_id(self) -> Optional[IMUDataIdentifier]:
        imu_value = self.imu_selector.currentData()
        if not imu_value:
            return None
        return IMUDataIdentifier(imu_value)

    def _filter_imu_ids(self, text: str) -> None:
        query = text.lower().strip()
        filtered = [
            entry for entry in self._imu_entries if query in self._entry_label(entry).lower()
        ]
        self._populate_selector(filtered)
        if not filtered:
            self._loaded_sensor_data = None
            self._loaded_imu_id = None
            self._clear_metadata()
            self._render_empty("No records match filter")
            return
        if self._loaded_imu_id is not None:
            selected_id = self._selected_imu_id()
            if selected_id is not None and selected_id != self._loaded_imu_id:
                self._loaded_sensor_data = None
                self._loaded_imu_id = None
                self._clear_metadata()
                self._render_empty(
                    "Selection changed. Click Load & Render to load this record."
                )

    def _on_imu_selection_changed(self, index: int) -> None:
        if index < 0:
            return
        selected_id = self._selected_imu_id()
        if selected_id is None:
            return
        if self._loaded_imu_id is not None and selected_id != self._loaded_imu_id:
            self._loaded_sensor_data = None
            self._loaded_imu_id = None
            self._clear_metadata()
            self._render_empty("Selection changed. Click Load & Render to load this record.")

    def _load_and_render(self) -> None:
        selected_id = self._selected_imu_id()
        if selected_id is None:
            self._render_empty("Select an IMU record first")
            return
        self._select_imu(selected_id)

    def _select_imu(self, imu_id: IMUDataIdentifier) -> None:
        if self._context is None:
            return
        imu_data = self._context.data_service.load_imu(imu_id)
        if not imu_data.data:
            self._loaded_sensor_data = None
            self._loaded_imu_id = imu_id
            self._render_empty("Selected IMU record has no sensor streams")
            return
        self._loaded_sensor_data = imu_data.data[0]
        self._loaded_imu_id = imu_id
        total_samples = len(self._loaded_sensor_data.time)
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setMaximum(max(0, total_samples - 1))
        self.end_spin.setMaximum(max(1, total_samples))
        self.start_spin.setValue(0)
        self.end_spin.setValue(total_samples)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)
        self._update_metadata(imu_id, imu_data.metadata.instrument_identifier.name)
        self._render_current()

    def _on_window_changed(self) -> None:
        if self.end_spin.value() <= self.start_spin.value():
            self.end_spin.setValue(self.start_spin.value() + 1)
        if self._loaded_sensor_data is not None:
            self._render_current()

    def _render_current(self) -> None:
        if self._context is None or self._loaded_sensor_data is None:
            return
        enabled_axes: Dict[str, bool] = {
            "vertical": self.axis_vertical.isChecked(),
            "mediolateral": self.axis_mediolateral.isChecked(),
            "anteroposterior": self.axis_anteroposterior.isChecked(),
        }
        if not any(enabled_axes.values()):
            self._render_empty("Enable at least one axis")
            return
        self._plot_engine.render(
            sensor_data=self._loaded_sensor_data,
            enabled_axes=enabled_axes,
            start_index=self.start_spin.value(),
            end_index=self.end_spin.value(),
            overlays=self._context.selection_state.state.overlays,
        )
        self._canvas.draw_idle()

    def _render_empty(self, message: str) -> None:
        self._figure.clear()
        axis = self._figure.add_subplot(111)
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _clear_metadata(self) -> None:
        self.meta_imu.setText("-")
        self.meta_user.setText("-")
        self.meta_instrument.setText("-")
        self.meta_spec.setText("-")

    def _update_metadata(self, imu_id: IMUDataIdentifier, instrument_name: str) -> None:
        if self._context is None:
            return
        self.meta_imu.setText(imu_id.value)
        user_data = self._context.data_service.load_user_for_imu(imu_id)
        self.meta_user.setText(user_data.user_identifier.value if user_data else "-")
        self.meta_instrument.setText(instrument_name or "-")
        spec_id, _spec = self._context.data_service.load_instrument_spec_for_imu(imu_id)
        self.meta_spec.setText(spec_id.value if spec_id else "-")
