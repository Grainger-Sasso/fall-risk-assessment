from pathlib import Path

from src.data_io.builders.model_builders.registry.registry_builder import (
    RegistryBuilder,
)
from src.database_manager.registry.registry import Registry
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import (
    RegistryHelper,
    TestConstants,
    TestSourceIdentifier,
)


class TestAggregateFeatureRegistryBuilder(BaseTest):
    def setUp(self):
        self.builder = RegistryBuilder()
        self.helper = RegistryHelper()

    def test_build(self):
        # Create test data
        test_data = self.helper.create_test_registry_csv()

        # Build registry
        result = self.builder.build(test_data, TestSourceIdentifier)

        # Verify result type
        self.assertIsInstance(result, Registry)

        # Verify registry contents
        for id, path in zip(
            TestConstants.REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(id, result.registry)
            self.assertEqual(result.registry[id], Path(path))

        self.assertEqual(result.id_type, TestSourceIdentifier)


if __name__ == "__main__":
    TestAggregateFeatureRegistryBuilder.run_tests()
