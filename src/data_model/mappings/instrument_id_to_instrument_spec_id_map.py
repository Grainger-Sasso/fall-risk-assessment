from dataclasses import dataclass
from typing import Dict

from src.data_model.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)
from src.data_model.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


@dataclass
class InstrumentIDToInstrumentSpecIDMap:
    map: Dict[InstrumentIdentifier:InstrumentSpecificationIdentifier]
