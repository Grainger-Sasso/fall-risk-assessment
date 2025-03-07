from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.database_manager.data_access.export_manager import ExportManager
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestImportManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()
        self.test_exporter = IMUDataFileExporter()
        self.manager = ExportManager({TestSourceIdentifier: self.test_exporter})

    def test_get_exporter(self):
        # Test successful exporter retrieval
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        exporter = self.manager.get_provider(type(source_id))
        self.assertIsInstance(exporter, IMUDataFileExporter)
        self.assertEqual(exporter, self.test_exporter)

    def test_invalid_data_type(self):
        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))


if __name__ == "__main__":
    TestImportManager.run_tests()
