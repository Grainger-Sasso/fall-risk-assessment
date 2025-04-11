from src.id_generator.identifier_generator import IdentifierGenerator
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


class InstrumentSpecificationIdentifierGenerator(IdentifierGenerator):
    def __init__(self):
        super().__init__(prefix="ins_spec", id_type=InstrumentSpecificationIdentifier)

    def generate_identifier(self) -> InstrumentSpecificationIdentifier:
        uuid = self._generate_uuid()
        id_value = self.prefix + "_" + str(uuid)
        return InstrumentSpecificationIdentifier(id_value)
