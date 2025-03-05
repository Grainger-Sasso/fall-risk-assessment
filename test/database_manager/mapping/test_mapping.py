from src.database_manager.mapping.mapping import Mapping
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestMapping(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()

        # Create test mapping using helper
        self.mapping: Mapping = self.helper.create_test_mapping()

    def test_get_target_id(self):
        # Test successful mapping
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        target_id = self.mapping.get_target_id(source_id)
        self.assertIsInstance(target_id, TestTargetIdentifier)
        self.assertEqual(target_id.value, TestConstants.TEST_TARGET_IDS.value[0])

    def test_invalid_id(self):
        # Test nonexistent mapping
        with self.assertRaises(KeyError):
            self.mapping.get_target_id(TestSourceIdentifier("nonexistent"))


if __name__ == "__main__":
    TestMapping.run_tests()
