from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.mapping.mapping import Mapping
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

    def test_get_mapping(self):
        # Test successful mapping
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        mapping = self.manager.get_provider(type(source_id))
        self.assertIsInstance(mapping, Mapping)
        self.assertEqual(mapping, self.test_mapping)

    def test_invalid_id(self):
        # Test nonexistent mapping
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))


if __name__ == "__main__":
    TestMappingManager.run_tests()
