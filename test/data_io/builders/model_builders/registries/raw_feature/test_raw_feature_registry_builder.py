from pathlib import Path

from src.data_io.builders.model_builders.registries.raw_feature.raw_feature_registry_builder import (
    RawFeatureRegistryBuilder,
)
from src.database_manager.registries.raw_feature.raw_feature_registry import (
    RawFeatureRegistry,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestRawFeatureRegistryBuilder(BaseTest):
    def setUp(self):
        self.builder = RawFeatureRegistryBuilder()
        self.helper = RegistryHelper()

    def test_build(self):
        # Create test data
        test_data = self.helper.create_test_registry_csv(
            TestConstants.RAW_FEATURE_REGISTRY_IDS.value
        )

        # Build registry
        result = self.builder.build(test_data)

        # Verify result type
        self.assertIsInstance(result, RawFeatureRegistry)

        # Verify registry contents
        for id, path in zip(
            TestConstants.RAW_FEATURE_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(RawFeatureIdentifier(id), result.registry)
            self.assertEqual(result.registry[RawFeatureIdentifier(id)], Path(path))


if __name__ == "__main__":
    TestRawFeatureRegistryBuilder.run_tests()
