from pathlib import Path

from src.data_io.builders.file_builders.registry.registry_file_builder import (
    RegistryFileBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.registry.registry_fields import RegistryFields
from src.data_model.registry.registry import Registry
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
)


class TestRegistryFileBuilder(BaseTest):
    def setUp(self):
        self.builder = RegistryFileBuilder()
        self.data_helper = DatabaseManagerTestHelper()

    def test_build_valid_registry(self):
        # Create test registry data
        registry_data: Registry = self.data_helper.create_test_registry()

        # Test building CSV file
        result: CSVFile = self.builder.build(registry_data)

        # Assertions
        self.assertIsInstance(result, CSVFile)

        # Test fieldnames
        expected_fieldnames = [
            RegistryFields.DATA_IDENTIFIER.value,
            RegistryFields.DIRECTORY.value,
        ]
        self.assertEqual(result.fieldnames, expected_fieldnames)

        # Test data
        self.assertIsInstance(result.data, dict)
        self.assertEqual(
            len(result.data[RegistryFields.DATA_IDENTIFIER.value]),
            len(TestConstants.TEST_SOURCE_IDS.value[:2]),
        )
        self.assertEqual(
            len(result.data[RegistryFields.DIRECTORY.value]),
            len(TestConstants.TEST_PATHS.value[:2]),
        )

        # Verify data values
        for idx, (id_val, path_val) in enumerate(
            zip(
                result.data[RegistryFields.DATA_IDENTIFIER.value],
                result.data[RegistryFields.DIRECTORY.value],
            )
        ):
            self.assertEqual(id_val, TestConstants.TEST_SOURCE_IDS.value[idx])
            self.assertEqual(path_val, str(TestConstants.TEST_PATHS.value[idx]))

    def test_build_empty_registry(self):
        # Create empty registry
        empty_registry = Registry(
            {}, id_type=TestSourceIdentifier, subdir_path=Path("/")
        )

        # Test building CSV file
        result = self.builder.build(empty_registry)

        # Verify empty data structure
        self.assertEqual(len(result.data[RegistryFields.DATA_IDENTIFIER.value]), 0)
        self.assertEqual(len(result.data[RegistryFields.DIRECTORY.value]), 0)


if __name__ == "__main__":
    TestRegistryFileBuilder.run_tests()
