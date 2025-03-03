from pathlib import Path

from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.registries.registry import Registry
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestIdentifier,
)


class TestRegistry(Registry):
    """Test implementation of Registry"""

    pass


class TestRegistryManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()

        # Create test registry using helper
        self.test_paths = self.helper.create_test_registry(
            TestConstants.TEST_SOURCE_IDS.value[:2], TestConstants.TEST_PATHS.value[:2]
        )
        self.registry = TestRegistry(self.test_paths)
        self.manager = RegistryManager[TestIdentifier](self.registry)

    def test_get_path(self):
        # Test successful path retrieval
        test_id = TestIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        path = self.manager.get_path(test_id)
        self.assertEqual(path, Path(TestConstants.TEST_PATHS.value[0]))

        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_path(TestIdentifier("nonexistent"))

    def test_get_all_paths(self):
        # Test getting all paths
        paths = self.manager.get_all_paths()
        self.assertEqual(paths, self.test_paths)
        self.assertEqual(len(paths), 2)


if __name__ == "__main__":
    TestRegistryManager.run_tests()
