from src.data_io.builders.model_builders.mappings.instrument_id_to_instrument_spec_id_map_builder import (
    InstrumentIDToInstrumentSpecIDMapBuilder,
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


class TestInstrumentIDToInstrumentSpecIDMapBuilder(BaseTest):
    def setUp(self):
        self.builder = InstrumentIDToInstrumentSpecIDMapBuilder()
        self.data_helper = MappingHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_mapping_csv(
            TestConstants.INST_TO_SPEC_SOURCE_IDS.value,
            TestConstants.INST_TO_SPEC_TARGET_IDS.value,
        )

        # Test building map
        result = self.builder.build(csv_data)

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

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestInstrumentIDToInstrumentSpecIDMapBuilder.run_tests()
