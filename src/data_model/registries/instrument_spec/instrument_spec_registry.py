from pathlib import Path
from typing import Dict

from src.data_model.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from src.data_model.registries.registry import Registry


class InstrumentSpecificationRegistry(Registry):
    def __init__(self, registry: Dict[InstrumentSpecificationIdentifier:Path]):
        super().__init__(registry)
