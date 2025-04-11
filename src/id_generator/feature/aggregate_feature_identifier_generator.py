from src.id_generator.identifier_generator import IdentifierGenerator
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)


class AggregateFeatureIdentifierGenerator(IdentifierGenerator):
    def __init__(self):
        super().__init__(prefix="agg", id_type=AggregateFeatureIdentifier)

    def generate_identifier(self) -> AggregateFeatureIdentifier:
        uuid = self._generate_uuid()
        id_value = self.prefix + "_" + str(uuid)
        return AggregateFeatureIdentifier(id_value)
