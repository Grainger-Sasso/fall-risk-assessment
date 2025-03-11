from pathlib import Path
from typing import Dict, List, Type

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.database_manager.mapping.mapping import Mapping
from src.identifiers.identifier import Identifier


class MappingBuilder(ModelBuilder):
    """Builds map of imu ID to user ID"""

    version = "1.0"

    def build(
        self,
        input_file: CSVFile,
        source_id_type: Type[Identifier],
        target_id_type: Type[Identifier],
        subdir_path: Path
    ) -> Mapping:
        if not isinstance(input_file, CSVFile):
            raise ValueError("Input must be an CSVFile")
        map: Dict[str, str] = {}
        source_ids: List[str] = [
            id for id in input_file.data[MappingFields.SOURCE_DATA_IDENTIFIER.value]
        ]
        target_ids: List[str] = [
            id for id in input_file.data[MappingFields.TARGET_DATA_IDENTIFIER.value]
        ]
        for s_id, t_id in zip(source_ids, target_ids):
            map[s_id] = t_id
        return Mapping(map, source_id_type, target_id_type, subdir_path)
