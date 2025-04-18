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
        path = self.registry.get_path_from_id(test_id)
        self.assertIsInstance(path, Path)
        self.assertEqual(path, TestConstants.TEST_PATHS.value[0])

    def test_add_entry(self):
        test_id: TestSourceIdentifier = TestSourceIdentifier('new_id')
        test_path: Path = Path("test/path")
        self.registry.add_entry(test_id, test_path)
        result_path = self.registry.get_path_from_id(test_id)
        self.assertIsInstance(result_path, Path)
        self.assertEqual(result_path, test_path)

    def test_update_entry(self):
        test_id: TestSourceIdentifier = self.helper.create_test_identifier()
        test_path: Path = Path("test/path")
        self.registry.update_entry(test_id, test_path)
        result_path = self.registry.get_path_from_id(test_id)
        self.assertIsInstance(result_path, Path)
        self.assertEqual(result_path, test_path)

    def test_invalid_id_type(self):
        # Test invalid identifier type
        with self.assertRaises(ValueError):
            self.registry.get_path_from_id(TestTargetIdentifier("nonexistent"))

    def test_nonexistent_id(self):
        with self.assertRaises(KeyError):
            self.registry.get_path_from_id(TestSourceIdentifier("nonexistent"))


if __name__ == "__main__":
    TestRegistry.run_tests()
