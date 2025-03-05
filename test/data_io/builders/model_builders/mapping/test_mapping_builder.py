from src.data_io.builders.model_builders.mapping.mapping_builder import MappingBuilder
from src.database_manager.mapping.mapping import Mapping
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import (
    MappingHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestMappingBuilder(BaseTest):
    def setUp(self):
        self.builder = MappingBuilder()
        self.data_helper = MappingHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_mapping_csv()

        # Test building map
        result = self.builder.build(
            csv_data, TestSourceIdentifier, TestTargetIdentifier
        )

        # Assertions
        self.assertIsInstance(result, Mapping)
        self.assertEqual(len(result.map), len(TestConstants.MAPPING_SOURCE_IDS.value))

        # Test mappings
        for ix, (source_id, target_id) in enumerate(result.map.items()):
            self.assertIsInstance(source_id, str)
            self.assertIsInstance(target_id, str)
            self.assertEqual(source_id, TestConstants.MAPPING_SOURCE_IDS.value[ix])
            self.assertEqual(target_id, TestConstants.MAPPING_TARGET_IDS.value[ix])

        self.assertEqual(result.source_id_type, TestSourceIdentifier)
        self.assertEqual(result.target_id_type, TestTargetIdentifier)

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None, None, None)


if __name__ == "__main__":
    TestMappingBuilder.run_tests()
