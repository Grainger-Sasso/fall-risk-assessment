from pathlib import Path
from typing import Dict, Type

from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.registries.registry import Registry
from src.identifiers.identifier import Identifier
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestIdentifier,
)


class TestRegistryManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()

        registry = self.helper.create_test_registry()
        registries: Dict[Type[Identifier], Registry] = {TestIdentifier: registry}
        self.manager = RegistryManager(registries)

    def test_get_path(self):
        # Test successful path retrieval
        test_id = TestIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        path = self.manager.get_path(test_id)
        self.assertEqual(path, Path(TestConstants.TEST_PATHS.value[0]))

        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_path(TestIdentifier("nonexistent"))


if __name__ == "__main__":
    TestRegistryManager.run_tests()
