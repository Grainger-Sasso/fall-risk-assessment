from src.database_manager.data_access.mapping_manager import MappingManager
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestIdentifier,
)


class TestMappingManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()

        # Create test mapping using helper
        self.test_mapping = self.helper.create_test_mapping(
            TestConstants.TEST_SOURCE_IDS.value[:2],
            TestConstants.TEST_TARGET_IDS.value[:2],
        )
        self.manager = MappingManager[TestIdentifier, TestIdentifier](self.test_mapping)

    def test_get_target_id(self):
        # Test successful mapping
        source_id = TestIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        target_id = self.manager.get_target_id(source_id)
        self.assertEqual(target_id, self.test_mapping[source_id])

        # Test nonexistent mapping
        with self.assertRaises(KeyError):
            self.manager.get_target_id(TestIdentifier("nonexistent"))

    def test_get_target_ids(self):
        # Test successful multiple mappings
        source_ids = [
            TestIdentifier(TestConstants.TEST_SOURCE_IDS.value[0]),
            TestIdentifier(TestConstants.TEST_SOURCE_IDS.value[1]),
        ]
        target_ids = self.manager.get_target_ids(source_ids)
        self.assertEqual(len(target_ids), 2)
        self.assertEqual(
            target_ids[0], TestIdentifier(TestConstants.TEST_TARGET_IDS.value[0])
        )
        self.assertEqual(
            target_ids[1], TestIdentifier(TestConstants.TEST_TARGET_IDS.value[1])
        )

        # Test with nonexistent mapping
        with self.assertRaises(KeyError):
            self.manager.get_target_ids(
                [
                    TestIdentifier("source_1"),
                    TestIdentifier("nonexistent"),
                ]
            )


if __name__ == "__main__":
    TestMappingManager.run_tests()
