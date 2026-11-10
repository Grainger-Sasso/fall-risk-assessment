import unittest

PYSIDE_AVAILABLE = True
try:
    from PySide6.QtWidgets import QApplication, QWidget
except Exception:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 not installed")
class TestVisualizationAppSmoke(unittest.TestCase):
    def setUp(self) -> None:
        from src.data_visualization.suite.plugins.base import PluginContext
        from src.data_visualization.suite.plugins.imu_record_explorer import (
            IMURecordExplorerPlugin,
        )
        from src.data_visualization.suite.state.selection_state_store import (
            SelectionStateStore,
        )

        class _FakeService:
            class _Repo:
                def list_record_ids(self, _id_type):
                    return []

                def get_record_path(self, _id_type, _record_id):
                    return "/tmp"

                def get_targets(self, _source_type, _source_id, relation_type=None):
                    return []

            def __init__(self):
                self.repository = self._Repo()

            def list_imu_ids(self):
                return []

            def get_user_for_imu(self, imu_id):
                return None

            def list_imu_records_with_users(self):
                return []

            def list_feature_ids(self):
                return []

            def load_imu(self, imu_id):
                raise RuntimeError("not used")

            def load_user_for_imu(self, imu_id):
                return None

            def load_instrument_spec_for_imu(self, imu_id):
                return None, None

        self.PluginContext = PluginContext
        self.SelectionStateStore = SelectionStateStore
        self.IMURecordExplorerPlugin = IMURecordExplorerPlugin
        self.fake_service = _FakeService()

    def test_main_window_initializes(self):
        from src.data_visualization.suite.app.main_window import VisualizationMainWindow
        from src.data_visualization.suite.services.visualization_data_service import (
            VisualizationDataService,
        )

        app = QApplication.instance() or QApplication([])
        window = VisualizationMainWindow(
            data_service=VisualizationDataService(db_manager=self.fake_service)
        )
        self.assertIsNotNone(window)
        window.close()
        if QApplication.instance() is app:
            app.quit()

    def test_imu_plugin_panel_builds(self):
        app = QApplication.instance() or QApplication([])
        plugin = self.IMURecordExplorerPlugin()
        context = self.PluginContext(
            data_service=self.fake_service,
            selection_state=self.SelectionStateStore(),
        )
        panel = plugin.build_panel(QWidget(), context)
        self.assertIsNotNone(panel)
        if QApplication.instance() is app:
            app.quit()

    def test_imu_plugin_lists_records_by_user(self):
        from src.data_visualization.suite.services.visualization_data_service import (
            IMURecordEntry,
        )
        from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
        from src.identifiers.user.user_identifier import UserIdentifier

        entries = [
            IMURecordEntry(IMUDataIdentifier("imu_b"), UserIdentifier("CO001")),
            IMURecordEntry(IMUDataIdentifier("imu_a"), UserIdentifier("FL002")),
            IMURecordEntry(IMUDataIdentifier("imu_c"), None),
        ]
        self.fake_service.list_imu_records_with_users = lambda: entries

        app = QApplication.instance() or QApplication([])
        plugin = self.IMURecordExplorerPlugin()
        context = self.PluginContext(
            data_service=self.fake_service,
            selection_state=self.SelectionStateStore(),
        )
        parent = QWidget()
        plugin.build_panel(parent, context)
        selector = plugin.imu_selector

        labels = [selector.itemText(i) for i in range(selector.count())]
        self.assertEqual(
            labels,
            ["CO001  —  imu_b", "FL002  —  imu_a", "(no user)  —  imu_c"],
        )
        self.assertEqual(selector.itemData(0), "imu_b")

        plugin.search_box.setText("fl002")
        self.assertEqual(selector.count(), 1)
        self.assertEqual(plugin._selected_imu_id().value, "imu_a")

        plugin.search_box.setText("imu_c")
        self.assertEqual(selector.count(), 1)
        self.assertEqual(selector.itemText(0), "(no user)  —  imu_c")
        if QApplication.instance() is app:
            app.quit()


if __name__ == "__main__":
    unittest.main()
