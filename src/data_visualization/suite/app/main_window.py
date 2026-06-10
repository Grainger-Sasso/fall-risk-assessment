from typing import Dict

from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from src.data_visualization.suite.plugins.base import PluginContext
from src.data_visualization.suite.plugins.feature_analytics_plugin import (
    FeatureAnalyticsPlugin,
)
from src.data_visualization.suite.plugins.imu_record_explorer import IMURecordExplorerPlugin
from src.data_visualization.suite.plugins.model_evaluation_plugin import (
    ModelEvaluationPlugin,
)
from src.data_visualization.suite.plugins.sql_index_inspector_plugin import (
    SQLIndexInspectorPlugin,
)
from src.data_visualization.suite.plugins.registry import VisualizationRegistry
from src.data_visualization.suite.services.visualization_data_service import (
    VisualizationDataService,
)
from src.data_visualization.suite.state.selection_state_store import SelectionStateStore


class VisualizationMainWindow(QMainWindow):
    def __init__(self, data_service: VisualizationDataService):
        super().__init__()
        self.setWindowTitle("Assessment Visualization Suite")
        self.resize(1400, 900)

        self.selection_state = SelectionStateStore()
        self.context = PluginContext(
            data_service=data_service,
            selection_state=self.selection_state,
        )
        self.registry = VisualizationRegistry()
        self.registry.register(IMURecordExplorerPlugin())
        self.registry.register(FeatureAnalyticsPlugin())
        self.registry.register(ModelEvaluationPlugin())
        self.registry.register(SQLIndexInspectorPlugin())

        self._plugin_order = self.registry.list_plugins()
        self._plugin_widgets: Dict[str, QWidget] = {}
        self._built_plugins: Dict[str, bool] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget(self)
        layout = QHBoxLayout(container)

        self.nav_list = QListWidget(container)
        self.nav_list.setMinimumWidth(260)
        self.nav_list.currentRowChanged.connect(self._on_plugin_selected)
        self.panel_stack = QStackedWidget(container)

        for plugin in self._plugin_order:
            self.nav_list.addItem(plugin.display_name)
            placeholder = QWidget(self.panel_stack)
            self._plugin_widgets[plugin.plugin_id] = placeholder
            self._built_plugins[plugin.plugin_id] = False
            self.panel_stack.addWidget(placeholder)

        layout.addWidget(self.nav_list, stretch=0)
        layout.addWidget(self.panel_stack, stretch=1)
        self.setCentralWidget(container)

    def _ensure_plugin_built(self, row: int) -> None:
        plugin = self._plugin_order[row]
        if self._built_plugins[plugin.plugin_id]:
            return
        placeholder = self._plugin_widgets[plugin.plugin_id]
        index = self.panel_stack.indexOf(placeholder)
        panel = plugin.build_panel(self.panel_stack, self.context)
        self.panel_stack.removeWidget(placeholder)
        placeholder.deleteLater()
        self.panel_stack.insertWidget(index, panel)
        self._plugin_widgets[plugin.plugin_id] = panel
        self._built_plugins[plugin.plugin_id] = True

    def _on_plugin_selected(self, row: int) -> None:
        if row < 0:
            return
        self._ensure_plugin_built(row)
        self.panel_stack.setCurrentIndex(row)
