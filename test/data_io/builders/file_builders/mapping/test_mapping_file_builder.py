from pathlib import Path

from src.data_io.builders.file_builders.mapping.mapping_file_builder import (
    MappingFileBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.data_model.mapping.mapping import Mapping
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestMappingFileBuilder(BaseTest):
    def setUp(self):
        self.builder = MappingFileBuilder()
        self.data_helper = DatabaseManagerTestHelper()

    def test_build_valid_mapping(self):
        # Create test mapping data
        mapping_data: Mapping = self.data_helper.create_test_mapping()

        # Test building CSV file
        result: CSVFile = self.builder.build(mapping_data)

        # Assertions
        self.assertIsInstance(result, CSVFile)

        # Test fieldnames
        expected_fieldnames = [
            MappingFields.SOURCE_DATA_IDENTIFIER.value,
            MappingFields.TARGET_DATA_IDENTIFIER.value,
        ]
        self.assertEqual(result.fieldnames, expected_fieldnames)

        # Test data
        self.assertIsInstance(result.data, dict)
        self.assertEqual(
            len(result.data[MappingFields.SOURCE_DATA_IDENTIFIER.value]),
            len(TestConstants.TEST_SOURCE_IDS.value),
        )
        self.assertEqual(
            len(result.data[MappingFields.TARGET_DATA_IDENTIFIER.value]),
            len(TestConstants.TEST_TARGET_IDS.value),
        )

        # Verify data values
        for idx, (source_id, target_id) in enumerate(
            zip(
                result.data[MappingFields.SOURCE_DATA_IDENTIFIER.value],
                result.data[MappingFields.TARGET_DATA_IDENTIFIER.value],
            )
        ):
            self.assertEqual(source_id, TestConstants.TEST_SOURCE_IDS.value[idx])
            self.assertEqual(target_id, TestConstants.TEST_TARGET_IDS.value[idx])

    def test_build_empty_mapping(self):
        # Create empty mapping
        empty_mapping = Mapping(
            {},
            source_id_type=TestSourceIdentifier,
            target_id_type=TestTargetIdentifier,
            subdir_path=Path("/"),
        )

        # Test building CSV file
        result = self.builder.build(empty_mapping)

        # Verify empty data structure
        self.assertEqual(
            len(result.data[MappingFields.SOURCE_DATA_IDENTIFIER.value]), 0
        )
        self.assertEqual(
            len(result.data[MappingFields.TARGET_DATA_IDENTIFIER.value]), 0
        )


if __name__ == "__main__":
    TestMappingFileBuilder.run_tests()
