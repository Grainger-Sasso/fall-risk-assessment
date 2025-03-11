from pathlib import Path
from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.database_manager.data_access.output_directory_manager import OutputDirectoryManager
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestOutputDirectoryManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()
        self.test_path = Path("/example/path")
        self.manager = OutputDirectoryManager({TestSourceIdentifier: self.test_path})

    def test_get_path(self):
        # Test successful exporter retrieval
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        path = self.manager.get_provider(type(source_id))
        self.assertIsInstance(path, Path)
        self.assertEqual(path, self.test_path)

    def test_invalid_data_type(self):
        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))


if __name__ == "__main__":
    TestOutputDirectoryManager.run_tests()
