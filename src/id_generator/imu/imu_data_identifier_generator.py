from src.id_generator.identifier_generator import IdentifierGenerator
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class IMUDataIdentifierGenerator(IdentifierGenerator):
    def __init__(self):
        super().__init__(prefix="imu", id_type=IMUDataIdentifier)

    def generate_identifier(self) -> IMUDataIdentifier:
        uuid = self._generate_uuid()
        id_value = self.prefix + "_" + str(uuid)
        return IMUDataIdentifier(id_value)
