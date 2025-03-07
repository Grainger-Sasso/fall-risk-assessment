from pathlib import Path
from typing import Dict, Type

from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestRegistryManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()
        self.registry = self.helper.create_test_registry()
        registries: Dict[Type[Identifier], Registry] = {
            self.registry.id_type: self.registry
        }
        self.manager = RegistryManager(registries)

    def test_get_registry(self):
        # Test successful registry retrieval
        test_id: TestSourceIdentifier = self.helper.create_test_identifier()
        registry = self.manager.get_provider(type(test_id))
        self.assertEqual(registry, self.registry)

    def test_invalid_id_type(self):
        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))


if __name__ == "__main__":
    TestRegistryManager.run_tests()
