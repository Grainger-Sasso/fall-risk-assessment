from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
    from src.data_visualization.suite.services.visualization_data_service import (
        VisualizationDataService,
    )
    from src.data_visualization.suite.state.selection_state_store import SelectionStateStore


@dataclass
class PluginContext:
    data_service: "VisualizationDataService"
    selection_state: "SelectionStateStore"


class VisualizationPlugin(ABC):
    plugin_id: str
    display_name: str

    @abstractmethod
    def build_panel(self, parent: "QWidget", context: PluginContext) -> "QWidget":
        """Build and return this plugin's root widget."""

    def on_selection_state_changed(self) -> None:
        """Optional hook for plugins with internal lifecycle behavior."""
