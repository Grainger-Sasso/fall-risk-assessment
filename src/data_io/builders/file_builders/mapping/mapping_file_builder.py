from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.data_model.mapping.mapping import Mapping


class MappingFileBuilder(FileBuilder):
    version = "1.0"

    def __init__(self):
        super().__init__()

    def build(self, data: Mapping) -> CSVFile:
        fieldnames = [
            MappingFields.SOURCE_DATA_IDENTIFIER.value,
            MappingFields.TARGET_DATA_IDENTIFIER.value,
        ]
        output_data = {
            MappingFields.SOURCE_DATA_IDENTIFIER.value: [],
            MappingFields.TARGET_DATA_IDENTIFIER.value: [],
        }
        for source_id, target_id in data.map.items():
            output_data[MappingFields.SOURCE_DATA_IDENTIFIER.value].append(source_id)
            output_data[MappingFields.TARGET_DATA_IDENTIFIER.value].append(target_id)
        return CSVFile(fieldnames=fieldnames, data=output_data)
