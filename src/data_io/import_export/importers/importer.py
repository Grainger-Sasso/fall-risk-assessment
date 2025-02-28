from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Generic, List, TypeVar

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.read_write.readers.file_reader import FileReader

# Generic type for the model object that will be built
T = TypeVar("T")


class Importer(Generic[T], ABC):
    """
    Generic interface for data importers.

    An importer combines a file reader and model builder to convert file data
    into model objects.
    """

    def __init__(
        self, reader: FileReader, model_builder: ModelBuilder, file_suffixes: List[str]
    ) -> None:
        """
        Initialize importer with a reader and model builder.

        Args:
            reader (FileReader): Reader for the specific file format
            model_builder (ModelBuilder): Builder for creating model objects
        """
        self.reader: FileReader = reader
        self.model_builder: ModelBuilder = model_builder
        self.file_suffixes: List[str] = file_suffixes

    @abstractmethod
    def import_data(self, directory: Path) -> T:
        """
        Import data from a file and convert it to a model object. Generally,
        the import data method will include the following steps:
        1. Resolve files from provided directory path to import
        2. Call readers to read the files into file objects
        3. Call builders to convert the file objects into data model objects

        Args:
            path (Path): Path to the directory of the data to import

        Returns:
            T: The constructed model object
        """
        pass

    def find_files_glob(self, directory: Path, suffix: str) -> List[Path]:
        """Find all files with given suffix in directory and subdirectories."""
        return list(directory.rglob(f"*.{suffix}"))

    def resolve_file_paths(
        self, directory: Path, file_names_enum: Enum
    ) -> Dict[Enum, Path]:
        """
        Resolve expected file paths from directory using enum of file names.

        Args:
            directory (Path): Directory to search
            file_names_enum (Enum): Enum class containing expected file names

        Returns:
            Dict[Enum, Path]: Mapping of enum members to their full file paths

        Raises:
            FileNotFoundError: If any expected file is not found
        """
        # Get all files with matching suffixes
        all_files = []
        for suffix in self.file_suffixes:
            all_files.extend(self.find_files_glob(directory, suffix))


        # Create mapping of enum members to paths
        file_paths: Dict[Enum, Path] = {}
        for enum_member in file_names_enum:
            # Find file whose stem (name without extension) matches enum value
            matching_file = next(
                (f for f in all_files if f.stem == enum_member.value), None
            )
            if matching_file is None:
                raise FileNotFoundError(
                    f"Could not find file with name {enum_member.value}"
                )
            file_paths[enum_member] = matching_file

        return file_paths
