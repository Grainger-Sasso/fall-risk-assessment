from pathlib import Path

from src.database_manager.registry.registry import Registry
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestRegistry(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()
        self.registry: Registry = self.helper.create_test_registry()

    def test_get_path(self):
        # Test successful registry retrieval
        test_id: TestSourceIdentifier = self.helper.create_test_identifier()
        path = self.registry.get_path(test_id)
        self.assertIsInstance(path, Path)
        self.assertEqual(path, TestConstants.TEST_PATHS.value[0])

    def test_invalid_id_type(self):
        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.registry.get_path(TestTargetIdentifier("nonexistent"))


if __name__ == "__main__":
    TestRegistry.run_tests()
