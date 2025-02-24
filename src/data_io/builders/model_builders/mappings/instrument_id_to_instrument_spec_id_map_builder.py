from typing import Dict, List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.database_manager.mappings.instrument_id_to_instrument_spec_id_map import (
    InstrumentIDToInstrumentSpecIDMap,
)
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


class InstrumentIDToInstrumentSpecIDMapBuilder(ModelBuilder):
    """Builds map of imu ID to user ID"""

    version = "1.0"

    def build(self, input_file: CSVFile) -> InstrumentIDToInstrumentSpecIDMap:
        map: Dict[InstrumentIdentifier, InstrumentSpecificationIdentifier] = {}
        instrument_ids: List[InstrumentIdentifier] = [
            InstrumentIdentifier(id.split("_")[0], id.split("_")[1])
            for id in input_file[MappingFields.SOURCE_DATA_IDENTIFIER]
        ]
        instrument_spec_ids: List[InstrumentSpecificationIdentifier] = [
            InstrumentSpecificationIdentifier(id)
            for id in input_file[MappingFields.TARGET_DATA_IDENTIFIER]
        ]
        for instrument_id, instrument_spec_id in zip(
            instrument_ids, instrument_spec_ids
        ):
            map[instrument_id] = instrument_spec_id
        return InstrumentIDToInstrumentSpecIDMap(map)
