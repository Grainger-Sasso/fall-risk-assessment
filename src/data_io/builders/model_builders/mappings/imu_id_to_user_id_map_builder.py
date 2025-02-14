from typing import Dict, List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.database_manager.mappings.imu_data_id_to_user_data_id_map import (
    IMUDataIDToUserDataIDMap,
)
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class IMUIDToUserIDMapBuilder(ModelBuilder):
    """Builds map of imu ID to user ID"""

    version = "1.0"

    def build(self, input_file: CSVFile) -> IMUDataIDToUserDataIDMap:
        map: Dict[IMUDataIdentifier:UserIdentifier] = {}
        imu_ids: List[IMUDataIdentifier] = [
            IMUDataIdentifier(id)
            for id in input_file[MappingFields.SOURCE_DATA_IDENTIFIER]
        ]
        user_ids: List[UserIdentifier] = [
            UserIdentifier(id)
            for id in input_file[MappingFields.TARGET_DATA_IDENTIFIER]
        ]
        for imu_id, user_id in zip(imu_ids, user_ids):
            map[imu_id] = user_id
        return IMUDataIDToUserDataIDMap(map)
