import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Tuple, TypeVar

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.file_format import FileFormat
from src.data_io.read_write.writers.file_writer import FileWriter
from src.identifiers.identifier import Identifier

# Type of data to be exported
T = TypeVar("T")


class Exporter(Generic[T], ABC):
    def __init__(
        self,
        file_builder: FileBuilder,
        writer: FileWriter,
        suffix: str,
        sub_dir_name: str,
    ):
        self.file_builder: FileBuilder = file_builder
        self.writer: FileWriter = writer
        self.suffix: str = suffix
        self.sub_dir_name: str = sub_dir_name

    def export_data(self, ouput_parent_dir: Path, data: T) -> Path:
        """Exports data model object to provided directory
        Args:
            directory (Path): parent output directory
            data (T):

        Returns:
            Path: path to output subdirectory
        """
        try:
            # 1. Get object ID for subdirectory
            obj_id: Identifier = self._get_object_id(data)

            # 2. Create output subdirectory
            output_subdir_path: Path = self._make_subdir(ouput_parent_dir, obj_id)

            # 3. Generate file object for export
            file_obj: FileFormat = self.file_builder.build(data)

            # 4. Write file object to path
            output_path: Path = self._construct_file_path(output_subdir_path)
            self.writer.write(output_path, file_obj)
            return output_subdir_path

        except Exception as e:
            raise Exception(f"Export failed: {str(e)}")

    def _make_subdir(self, output_parent_dir: Path, obj_id: Identifier) -> Path:
        try:
            output_subdir_name = self.sub_dir_name + obj_id.value
            output_subdir_path = os.path.join(output_parent_dir, output_subdir_name)
            os.makedirs(output_subdir_path, exist_ok=True)
            return Path(output_subdir_path)
        except PermissionError:
            # Occurs when user doesn't have permission to create directory
            raise PermissionError(
                f"Permission denied: Unable to create directory at {output_subdir_path}"
            )
        except OSError as e:
            # Catches other OS-related errors (disk full, invalid characters in path, etc.)
            raise OSError(
                f"Failed to create directory at {output_subdir_path}: {str(e)}"
            )

    @abstractmethod
    def _get_object_id(self, data: T) -> Identifier:
        pass

    @abstractmethod
    def _construct_file_path(self, output_subdir_path: Path) -> Path:
        pass
