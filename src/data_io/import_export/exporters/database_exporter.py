import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.file_format import FileFormat
from src.data_io.read_write.writers.file_writer import FileWriter

T = TypeVar("T")


class DatabaseExporter(Generic[T], ABC):
    def __init__(self, file_builder: FileBuilder, writer: FileWriter, suffix: str):
        self.file_builder: FileBuilder = file_builder
        self.writer: FileWriter = writer
        self.suffix: str = suffix

    def export_data(self, output_dir: Path, data: T) -> Path:
        try:
            # Construct file object
            # file_data: FileFormat = self.file_builder.build(data)
            csv_file: FileFormat = self.file_builder.build(data)
            # Check directory for file in the input path
            file_name: str = self._get_file_name()
            full_file_name = file_name + "." + self.suffix
            full_file_path = output_dir / full_file_name
            copy = False
            if self._file_exists(full_file_path):
                # If file exists, create copy and delete existing file
                copy_path: Path = self._copy_existing_file(output_dir, full_file_path)
                self._delete_file(full_file_path)
                copy = True
            # Write new file, delete copy if successful
            self.writer.write(full_file_path, csv_file)
            if copy:
                self._delete_file(copy_path)
            return full_file_path

        except Exception as e:
            raise Exception(f"Export failed: {str(e)}")

    @abstractmethod
    def _get_file_name(self) -> str:
        pass

    def _file_exists(self, file_path: Path) -> bool:
        return file_path.exists()

    def _copy_existing_file(self, output_dir: Path, file_path: Path) -> Path:
        try:
            # Get original name and add "copy"
            new_name = f"{file_path.stem}_copy{file_path.suffix}"
            output_path = output_dir / new_name
            # Copy the file
            shutil.copy2(file_path, output_path)
            return output_path

        except OSError as e:
            raise OSError(f"Failed to copy {file_path} to {output_dir}: {str(e)}")

    def _delete_file(self, file_path: Path):
        try:
            file_path.unlink()

        except PermissionError:
            raise PermissionError(f"Permission denied deleting file: {file_path}")
        except OSError as e:
            raise OSError(f"Failed to delete file {file_path}: {str(e)}")
