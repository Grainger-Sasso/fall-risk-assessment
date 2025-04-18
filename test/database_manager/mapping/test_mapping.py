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
        target_id = self.mapping.get_target_id_from_source_id(source_id)
        self.assertIsInstance(target_id, TestTargetIdentifier)
        self.assertEqual(target_id.value, TestConstants.TEST_TARGET_IDS.value[0])

    def test_add_entry(self):
        test_source_id: TestSourceIdentifier = TestSourceIdentifier('new_source_id')
        test_target_id: TestTargetIdentifier = TestTargetIdentifier('new_target_id')
        self.mapping.add_entry(test_source_id, test_target_id)
        result_id = self.mapping.get_target_id_from_source_id(test_source_id)
        self.assertIsInstance(result_id, TestTargetIdentifier)
        self.assertEqual(result_id, test_target_id)

    def test_update_entry(self):
        test_source_id: TestSourceIdentifier = self.helper.create_test_identifier()
        test_target_id: TestTargetIdentifier = TestTargetIdentifier('new_target_id')
        self.mapping.update_entry(test_source_id, test_target_id)
        result_id = self.mapping.get_target_id_from_source_id(test_source_id)
        self.assertIsInstance(result_id, TestTargetIdentifier)
        self.assertEqual(result_id, test_target_id)

    def test_invalid_source_id_type(self):
        # Test invalid source ID type 
        with self.assertRaises(ValueError):
            self.mapping.get_target_id_from_source_id(
                TestTargetIdentifier("nonexistent")
            )
    
    def test_nonexistent_source_id(self):
        # Test nonexistent source ID
        with self.assertRaises(KeyError):
            self.mapping.get_target_id_from_source_id(
                TestSourceIdentifier("nonexistent")
            )


if __name__ == "__main__":
    TestMapping.run_tests()
