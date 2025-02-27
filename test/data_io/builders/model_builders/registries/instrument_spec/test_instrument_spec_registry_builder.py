from pathlib import Path

from src.data_io.builders.model_builders.registries.instrument_spec.instrument_spec_registry_builder import (
    InstrumentSpecificationRegistryBuilder,
)
from src.database_manager.registries.instrument_spec.instrument_spec_registry import (
    InstrumentSpecificationRegistry,
)
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestInstrumentSpecificationRegistryBuilder(BaseTest):
    def setUp(self):
        self.builder = InstrumentSpecificationRegistryBuilder()
        self.helper = RegistryHelper()

    def test_build(self):
        # Create test data
        test_data = self.helper.create_test_registry_csv(
            TestConstants.INSTRUMENT_SPEC_REGISTRY_IDS.value
        )

        # Build registry
        result = self.builder.build(test_data)

        # Verify result type
        self.assertIsInstance(result, InstrumentSpecificationRegistry)

        # Verify registry contents
        for id, path in zip(
            TestConstants.INSTRUMENT_SPEC_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(InstrumentSpecificationIdentifier(id), result.registry)
            self.assertEqual(
                result.registry[InstrumentSpecificationIdentifier(id)], Path(path)
            )


if __name__ == "__main__":
    TestInstrumentSpecificationRegistryBuilder.run_tests() 