from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.registries.registry_builder import (
    RegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.database_manager.registries.instrument_spec.instrument_spec_registry import (
    InstrumentSpecificationRegistry,
)
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


class InstrumentSpecificationRegistryBuilder(RegistryBuilder):
    version = "1.0"

    def build(self, input_file: CSVFile) -> InstrumentSpecificationRegistry:
        registry: Dict[str, Path] = self.build_registry(input_file)
        return InstrumentSpecificationRegistry(
            {
                InstrumentSpecificationIdentifier(id): path
                for id, path in registry.items()
            }
        )
