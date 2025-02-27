from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.data.user.user_data_builder import (
    UserDataBuilder,
)
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.json.json_dict_file_reader import JSONDictFileReader
from src.data_model.data.user.user_data import UserData


class UserDataFileNames(Enum):
    USER_DATA = "user_data"
    CLININCAL_DEMOGRAPHIC_DATA = "clinical_demographic_data"


class UserDataImporter(Importer[UserData]):
    """Importer for user data from JSON files."""

    def __init__(self) -> None:
        reader = JSONDictFileReader()
        model_builder = UserDataBuilder()
        file_suffixes = ["json"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> UserData:
        # Resolve paths for all required files
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, UserDataFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )

        # Read files using resolved paths
        user_data_file: JSONDictFile = self.reader.read(
            file_paths[UserDataFileNames.USER_DATA]
        )
        clinical_demo_data_file: JSONDictFile = self.reader.read(
            file_paths[UserDataFileNames.CLININCAL_DEMOGRAPHIC_DATA]
        )

        # Build model from file data
        return self.model_builder.build(user_data_file, clinical_demo_data_file)
