from typing import Dict, List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.database_manager.mappings.aggregate_feature_id_to_raw_feature_id_map import (
    AggregateFeatureIDToRawFeatureIDMap,
)
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import (
    RawFeatureIdentifier,
)


class AggregateIDToRawIDMapBuilder(ModelBuilder):
    """Builds map of aggregate ID to raw ID"""

    version = "1.0"

    def build(self, input_file: CSVFile) -> AggregateFeatureIDToRawFeatureIDMap:
        map: Dict[AggregateFeatureIdentifier:RawFeatureIdentifier] = {}
        aggregate_ids: List[AggregateFeatureIdentifier] = [
            AggregateFeatureIdentifier(id)
            for id in input_file[MappingFields.SOURCE_DATA_IDENTIFIER]
        ]
        raw_ids: List[RawFeatureIdentifier] = [
            RawFeatureIdentifier(id)
            for id in input[MappingFields.TARGET_DATA_IDENTIFIER]
        ]
        for aggregate_id, raw_id in zip(aggregate_ids, raw_ids):
            map[aggregate_id] = raw_id
        return AggregateFeatureIDToRawFeatureIDMap(map)
