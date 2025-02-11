import csv
from pathlib import Path
from typing import List, Dict, Any
from src.data_io.read_write.readers.file_reader import FileReader


class CSVFileReader(FileReader):
    """
    CSV file reader
    """

    def read(self, path: Path) -> List[Dict[str, Any]]:
        """Reads dsata from a CSV file and returns it as a list of dictionaries.

        Args:
            path (Path): The path to the CSV file.

        Returns:
            List[Dict[str, Any]]: A list of rows, where each row is a dictionary
                                   mapping column names to values.
        """
        if not path.exists():
            raise FileNotFoundError(f"The file at {path} does not exist.")

        if not path.suffix == ".csv":
            raise ValueError(f"Expected a .csv file, but got {path.suffix}.")

        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)
