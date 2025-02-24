from typing import Dict, List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.database_manager.mappings.raw_feature_id_to_imu_data_id_map import (
    RawFeatureIDToIMUDataIDMap,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class RawFeatureIDToIMUIDMapBuilder(ModelBuilder):
    """Builds map of imu ID to user ID"""

    version = "1.0"

    def build(self, input_file: CSVFile) -> RawFeatureIDToIMUDataIDMap:
        map: Dict[RawFeatureIdentifier, IMUDataIdentifier] = {}
        raw_ids: List[RawFeatureIdentifier] = [
            RawFeatureIdentifier(id)
            for id in input_file[MappingFields.SOURCE_DATA_IDENTIFIER]
        ]
        imu_ids: List[IMUDataIdentifier] = [
            IMUDataIdentifier(id)
            for id in input_file[MappingFields.TARGET_DATA_IDENTIFIER]
        ]
        for raw_id, imu_id in zip(raw_ids, imu_ids):
            map[raw_id] = imu_id
        return RawFeatureIDToIMUDataIDMap(map)
