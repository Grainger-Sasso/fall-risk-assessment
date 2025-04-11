from src.id_generator.identifier_generator import IdentifierGenerator
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier


class RawFeatureIdentifierGenerator(IdentifierGenerator):
    def __init__(self):
        super().__init__(prefix="raw", id_type=RawFeatureIdentifier)

    def generate_identifier(self) -> RawFeatureIdentifier:
        uuid = self._generate_uuid()
        id_value = self.prefix + "_" + str(uuid)
        return RawFeatureIdentifier(id_value)
