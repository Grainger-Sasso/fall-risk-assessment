from typing import Dict

from src.database_manager.mappings.mapping import Mapping
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


class InstrumentIDToInstrumentSpecIDMap(
    Mapping[InstrumentIdentifier, InstrumentSpecificationIdentifier]
):
    """Maps instrument IDs to instrument specification IDs"""

    def __init__(
        self,
        map: Dict[InstrumentIdentifier, InstrumentSpecificationIdentifier],
    ):
        super().__init__(map)
