from src.id_generator.identifier_generator import IdentifierGenerator
from src.identifiers.user.user_identifier import UserIdentifier


class UserIdentifierGenerator(IdentifierGenerator):
    def __init__(self):
        super().__init__(prefix="usr", id_type=UserIdentifier)

    def generate_identifier(self) -> UserIdentifier:
        uuid = self._generate_uuid()
        id_value = self.prefix + "_" + str(uuid)
        return UserIdentifier(id_value)
