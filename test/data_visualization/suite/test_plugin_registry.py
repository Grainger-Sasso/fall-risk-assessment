import unittest

from src.data_visualization.suite.plugins.base import VisualizationPlugin
from src.data_visualization.suite.plugins.registry import VisualizationRegistry


class _DummyPlugin(VisualizationPlugin):
    plugin_id = "dummy"
    display_name = "Dummy"

    def build_panel(self, parent, context):
        return None


class TestVisualizationRegistry(unittest.TestCase):
    def test_register_and_get_plugin(self):
        registry = VisualizationRegistry()
        plugin = _DummyPlugin()
        registry.register(plugin)
        fetched = registry.get("dummy")
        self.assertEqual(fetched.display_name, "Dummy")

    def test_duplicate_registration_raises(self):
        registry = VisualizationRegistry()
        registry.register(_DummyPlugin())
        with self.assertRaises(ValueError):
            registry.register(_DummyPlugin())


if __name__ == "__main__":
    unittest.main()
