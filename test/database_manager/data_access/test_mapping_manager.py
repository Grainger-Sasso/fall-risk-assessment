from src.database_manager.data_access.mapping_manager import MappingManager
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestMappingManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()

        # Create test mapping using helper
        self.test_mapping = self.helper.create_test_mapping()
        self.manager = MappingManager({TestSourceIdentifier: self.test_mapping})

    def test_get_target_id(self):
        # Test successful mapping
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        target_id = self.manager.get_target_id(source_id)
        self.assertIsInstance(target_id, TestTargetIdentifier)
        self.assertEqual(target_id.value, TestConstants.TEST_TARGET_IDS.value[0])

        # Test nonexistent mapping
        with self.assertRaises(KeyError):
            self.manager.get_target_id(TestSourceIdentifier("nonexistent"))

        with self.assertRaises(KeyError):
            self.manager.get_target_id(TestTargetIdentifier("nonexistent"))


if __name__ == "__main__":
    TestMappingManager.run_tests()
