from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataImporter,
)
from src.database_manager.data_access.import_manager import ImportManager
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
        self.test_importer = IMUDataImporter()
        self.manager = ImportManager({TestSourceIdentifier: self.test_importer})

    def test_get_importer(self):
        # Test successful importer retrieval
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        importer = self.manager.get_provider(type(source_id))
        self.assertIsInstance(importer, IMUDataImporter)
        self.assertEqual(importer, self.test_importer)

    def test_invalid_data_type(self):
        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))


if __name__ == "__main__":
    TestImportManager.run_tests()
