from enum import Enum
from pathlib import Path
from typing import Dict

from src.data_io.builders.model_builders.dataset.dataset_builder import DatasetBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.importers.importer import Importer
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.data_model.dataset.dataset import Dataset


class DatasetFileNames(Enum):
    """Enumeration of dataset file names."""

    DATASET = "dataset"


class DatasetImporter(Importer[Dataset]):
    """Imports dataset data from files."""

    def __init__(self):
        reader = CSVFileReader()
        model_builder = DatasetBuilder()
        file_suffixes = ["csv"]
        super().__init__(reader, model_builder, file_suffixes)

    def import_data(self, directory: Path, dataset_name: str) -> Dataset:
        """Import dataset data from directory.

        Args:
            directory (Path): Directory containing dataset files
            dataset_name (str): Name of dataset to import

        Returns:
            Dataset: Imported dataset

        Raises:
            FileNotFoundError: If required files not found
        """
        file_paths: Dict[Enum, Path] = self.resolve_file_paths(
            directory, DatasetFileNames
        )
        if not all([p.exists for _, p in file_paths.items()]):
            raise (
                FileNotFoundError(
                    f"Files missing in set of file paths found: {file_paths}"
                )
            )
        dataset_file: CSVFile = self.reader.read(file_paths[DatasetFileNames.DATASET])
        return self.model_builder.build(dataset_file, dataset_name)
