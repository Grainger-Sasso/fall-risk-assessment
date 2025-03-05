from typing import Dict, List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.database_manager.mapping.mapping import Mapping
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class MappingBuilder(ModelBuilder):
    """Builds map of imu ID to user ID"""

    version = "1.0"

    def build(self, input_file: CSVFile) -> Mapping:
        if not isinstance(input_file, CSVFile):
            raise ValueError("Input must be an CSVFile")
        map: Dict[str, str] = {}
        source_ids: List[IMUDataIdentifier] = [
            id for id in input_file.data[MappingFields.SOURCE_DATA_IDENTIFIER.value]
        ]
        target_ids: List[UserIdentifier] = [
            id for id in input_file.data[MappingFields.TARGET_DATA_IDENTIFIER.value]
        ]
        for s_id, t_id in zip(source_ids, target_ids):
            map[s_id] = t_id
        return Mapping(map)
