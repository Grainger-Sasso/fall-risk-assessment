from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.instrument_specifications.instrument_specification_builder import (
    IMUSpecificationBuilder,
)
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.json.json_dict_file_reader import JSONDictFileReader
from src.data_model.instrument_specifications.imu_specifications import (
    IMUSpecifications,
)


class InstrumentSpecificationFileNames(Enum):
    INSTRUMENT_SPEC = "instrument_spec"


class InstrumentSpecificationImporter(Importer[IMUSpecifications]):
    """Importer for user data from JSON files."""

    def __init__(self) -> None:
        reader = JSONDictFileReader()
        model_builder = IMUSpecificationBuilder()
        file_suffixes = ["json"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path) -> IMUSpecifications:
        # Resolve paths for all required files
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, InstrumentSpecificationFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )

        # Read files using resolved paths
        instrument_specification_file: JSONDictFile = self.reader.read(
            file_paths[InstrumentSpecificationFileNames.INSTRUMENT_SPEC]
        )

        # Build model from file data
        return self.model_builder.build(instrument_specification_file)
