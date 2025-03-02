import tempfile
from pathlib import Path

from src.data_io.import_export.importers.mappings.instrument_id_to_instrument_spec_id_map_importer import (
    InstrumentIDToInstrumentSpecIDMapFileNames,
    InstrumentIDToInstrumentSpecIDMapImporter,
)
from src.database_manager.mappings.instrument_id_to_instrument_spec_id_map import (
    InstrumentIDToInstrumentSpecIDMap,
)
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import MappingHelper, TestConstants


class TestInstrumentIDToInstrumentSpecIDMapImporter(BaseTest):
    def setUp(self):
        self.importer = InstrumentIDToInstrumentSpecIDMapImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = MappingHelper()

        # Create test file in temp directory
        self.map_path = self.helper.create_test_mapping_file(
            TestConstants.INST_TO_SPEC_SOURCE_IDS.value,
            TestConstants.INST_TO_SPEC_TARGET_IDS.value,
            self.temp_path
            / f"{InstrumentIDToInstrumentSpecIDMapFileNames.INSTRUMENT_TO_INSTRUMENT_SPEC_ID_MAP.value}.csv",
        )

    def tearDown(self):
        # Clean up test files
        if self.map_path.exists():
            self.map_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Assertions
        self.assertIsInstance(result, InstrumentIDToInstrumentSpecIDMap)
        self.assertEqual(
            len(result.map), len(TestConstants.INST_TO_SPEC_SOURCE_IDS.value)
        )

        # Test mappings
        for ix, (inst_id, spec_id) in enumerate(result.map.items()):
            self.assertIsInstance(inst_id, InstrumentIdentifier)
            self.assertIsInstance(spec_id, InstrumentSpecificationIdentifier)

            # Split source ID into manufacturer and serial number
            manufacturer, serial = TestConstants.INST_TO_SPEC_SOURCE_IDS.value[
                ix
            ].split("_")
            self.assertEqual(inst_id.name, manufacturer)
            self.assertEqual(inst_id.serial_number, serial)

            self.assertEqual(
                spec_id.value, TestConstants.INST_TO_SPEC_TARGET_IDS.value[ix]
            )

    def test_missing_file(self):
        # Remove the required file
        self.map_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestInstrumentIDToInstrumentSpecIDMapImporter.run_tests()
