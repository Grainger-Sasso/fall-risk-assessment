import json
from pathlib import Path
from typing import Any
from src.data_io.read_write.readers.file_reader import FileReader


class JSONFileReader(FileReader):
    """
    JSON file reader
    """

    def read(self, path: Path) -> Any:
        """Reads and parses data from a JSON file.

        Args:
            path (Path): The path to the JSON file.

        Returns:
            Any: The parsed data (could be a dict, list, etc.).
        """
        if not path.exists():
            raise FileNotFoundError(f"The file at {path} does not exist.")

        if not path.suffix == ".json":
            raise ValueError(f"Expected a .json file, but got {path.suffix}.")

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
