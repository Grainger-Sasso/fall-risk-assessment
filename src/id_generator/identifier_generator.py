import uuid
from abc import ABC

from src.identifiers.identifier import Identifier


class IdentifierGenerator(ABC):
    def __init__(self, prefix: str, id_type: Identifier):
        self.prefix: str = prefix
        self.id_type: Identifier = id_type

    def generate_identifier(self) -> Identifier:
        uuid = self._generate_uuid()
        id_value = self.prefix + "_" + uuid
        return Identifier(id_value)

    def _generate_uuid(self) -> str:
        return uuid.uuid4()
