import csv
from pathlib import Path
from typing import Any, Tuple

from src.data_io.read_write.writers.file_writer import FileWriter
from src.data_io.formats.csv.csv_file import CSVFile


class CSVFileWriter(FileWriter):
    """
    CSV file writer.
    """

    def write(self, path: Path, data: CSVFile, **kwargs) -> Tuple[bool, str]:
        """
        Writes CSV data to a file.

        Args:
            path (Path): File path to write.
            data (CSVFile): Data to write.
            **kwargs: Additional parameters for specific file formats (e.g., compression, delimiter).

        Returns:
            Tuple[bool, str]: (success flag, error message)
        """
        try:
            with open(path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=CSVFile.fieldnames, **kwargs)
                writer.writeheader()
                writer.writerows(CSVFile.data)
            return True, ""
        except Exception as e:
            return False, str(e)
