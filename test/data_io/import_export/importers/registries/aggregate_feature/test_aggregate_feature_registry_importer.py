import tempfile
from pathlib import Path

from src.data_io.import_export.importers.registries.aggregate_feature.aggregate_feature_registry_importer import (
    AggregateFeatureRegistryFileNames,
    AggregateFeatureRegistryImporter,
)
from src.database_manager.registries.aggregate_feature.aggregate_feature_registry import (
    AggregateFeatureRegistry,
)
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestAggregateFeatureRegistryImporter(BaseTest):
    def setUp(self):
        self.importer = AggregateFeatureRegistryImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = RegistryHelper()

        # Create test file in temp directory
        self.registry_path = self.helper.create_test_registry_file(
            TestConstants.AGG_FEATURE_REGISTRY_IDS.value,
            self.temp_path
            / f"{AggregateFeatureRegistryFileNames.AGGREGATE_FEATURE_REGISTRY.value}.csv",
        )

    def tearDown(self):
        # Clean up test files
        if self.registry_path.exists():
            self.registry_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Assertions
        self.assertIsInstance(result, AggregateFeatureRegistry)
        self.assertEqual(
            len(result.registry), len(TestConstants.AGG_FEATURE_REGISTRY_IDS.value)
        )

        # Test registry entries
        for id, path in zip(
            TestConstants.AGG_FEATURE_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(AggregateFeatureIdentifier(id), result.registry)
            self.assertEqual(
                result.registry[AggregateFeatureIdentifier(id)], Path(path)
            )

    def test_missing_file(self):
        # Remove the required file
        self.registry_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestAggregateFeatureRegistryImporter.run_tests()
