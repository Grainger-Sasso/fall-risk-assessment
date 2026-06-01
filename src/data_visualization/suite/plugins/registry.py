from dataclasses import dataclass
from typing import Dict, List

from src.data_visualization.suite.plugins.base import VisualizationPlugin


@dataclass
class VisualizationRegistry:
    _plugins: Dict[str, VisualizationPlugin]

    def __init__(self) -> None:
        self._plugins = {}

    def register(self, plugin: VisualizationPlugin) -> None:
        if plugin.plugin_id in self._plugins:
            raise ValueError(f"Plugin already registered: {plugin.plugin_id}")
        self._plugins[plugin.plugin_id] = plugin

    def get(self, plugin_id: str) -> VisualizationPlugin:
        if plugin_id not in self._plugins:
            raise KeyError(f"Unknown plugin: {plugin_id}")
        return self._plugins[plugin_id]

    def list_plugins(self) -> List[VisualizationPlugin]:
        return list(self._plugins.values())
