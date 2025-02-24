from dataclasses import dataclass
from typing import Dict

from src.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


@dataclass
class InstrumentIDToInstrumentSpecIDMap:
    map: Dict[InstrumentIdentifier, InstrumentSpecificationIdentifier]
