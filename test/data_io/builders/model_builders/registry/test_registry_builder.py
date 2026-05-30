from pathlib import Path

from src.data_io.builders.model_builders.registry.registry_builder import (
    RegistryBuilder,
)
from src.data_model.registry.registry import Registry
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
        self.test_path = Path("/test/path")

    def test_build(self):
        # Create test data
        test_data = self.helper.create_test_registry_csv()

        # Build registry
        result = self.builder.build(test_data, TestSourceIdentifier, self.test_path)

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

        self.assertIsInstance(result.path, Path)
        self.assertEqual(result.path, self.test_path)


if __name__ == "__main__":
    TestAggregateFeatureRegistryBuilder.run_tests()
