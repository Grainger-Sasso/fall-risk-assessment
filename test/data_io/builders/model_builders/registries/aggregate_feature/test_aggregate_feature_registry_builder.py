from pathlib import Path

from src.data_io.builders.model_builders.registries.aggregate_feature.aggregate_feature_registry_builder import (
    AggregateFeatureRegistryBuilder,
)
from src.database_manager.registries.aggregate_feature.aggregate_feature_registry import (
    AggregateFeatureRegistry,
)
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestAggregateFeatureRegistryBuilder(BaseTest):
    def setUp(self):
        self.builder = AggregateFeatureRegistryBuilder()
        self.helper = RegistryHelper()

    def test_build(self):
        # Create test data
        test_data = self.helper.create_test_registry_csv(
            TestConstants.AGG_FEATURE_REGISTRY_IDS.value
        )

        # Build registry
        result = self.builder.build(test_data)

        # Verify result type
        self.assertIsInstance(result, AggregateFeatureRegistry)

        # Verify registry contents
        for id, path in zip(
            TestConstants.AGG_FEATURE_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(AggregateFeatureIdentifier(id), result.registry)
            self.assertEqual(
                result.registry[AggregateFeatureIdentifier(id)], Path(path)
            )


if __name__ == "__main__":
    TestAggregateFeatureRegistryBuilder.run_tests()
