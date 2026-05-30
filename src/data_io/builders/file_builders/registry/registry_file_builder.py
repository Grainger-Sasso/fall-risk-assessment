from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.model_fields.registry.registry_fields import RegistryFields
from src.data_model.registry.registry import Registry


class RegistryFileBuilder(FileBuilder):
    version = "1.0"

    def __init__(self):
        super().__init__()

    def build(self, data: Registry) -> CSVFile:
        fieldnames = [
            RegistryFields.DATA_IDENTIFIER.value,
            RegistryFields.DIRECTORY.value,
        ]
        output_data = {
            RegistryFields.DATA_IDENTIFIER.value: [],
            RegistryFields.DIRECTORY.value: [],
        }
        for source_id, directory in data.registry.items():
            output_data[RegistryFields.DATA_IDENTIFIER.value].append(source_id)
            output_data[RegistryFields.DIRECTORY.value].append(str(directory))
        return CSVFile(fieldnames=fieldnames, data=output_data)
