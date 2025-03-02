import tempfile
from pathlib import Path

from src.data_io.import_export.importers.registries.instrument_spec.instrument_spec_registry_importer import (
    InstrumentSpecificationRegistryFileNames,
    InstrumentSpecificationRegistryImporter,
)
from src.database_manager.registries.instrument_spec.instrument_spec_registry import (
    InstrumentSpecificationRegistry,
)
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestInstrumentSpecificationRegistryImporter(BaseTest):
    def setUp(self):
        self.importer = InstrumentSpecificationRegistryImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = RegistryHelper()

        # Create test file in temp directory
        self.registry_path = self.helper.create_test_registry_file(
            TestConstants.INSTRUMENT_SPEC_REGISTRY_IDS.value,
            self.temp_path
            / f"{InstrumentSpecificationRegistryFileNames.INSTRUMENT_SPEC_REGISTRY.value}.csv",
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
        self.assertIsInstance(result, InstrumentSpecificationRegistry)
        self.assertEqual(
            len(result.registry), len(TestConstants.INSTRUMENT_SPEC_REGISTRY_IDS.value)
        )

        # Test registry entries
        for id, path in zip(
            TestConstants.INSTRUMENT_SPEC_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(InstrumentSpecificationIdentifier(id), result.registry)
            self.assertEqual(
                result.registry[InstrumentSpecificationIdentifier(id)], Path(path)
            )

    def test_missing_file(self):
        # Remove the required file
        self.registry_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestInstrumentSpecificationRegistryImporter.run_tests()
