import csv
from pathlib import Path
from typing import Any, Tuple

from src.data_io.read_write.writers.file_writer import FileWriter


class CSVFileWriter(FileWriter):
    """
    CSV file writer.
    """

    def write(self, path: Path, data: Any, **kwargs) -> Tuple[bool, str]:
        """
        Writes CSV data to a file.

        Args:
            path (Path): File path to write.
            data (DataType): Data to write. Expected to be a dictionary or list of dictionaries.
            **kwargs: Additional parameters for specific file formats (e.g., compression, delimiter).

        Returns:
            Tuple[bool, str]: (success flag, error message)
        """
        try:
            with open(path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=data[0].keys(), **kwargs)
                writer.writeheader()
                writer.writerows(data)
            return True, ""
        except Exception as e:
            return False, str(e)
