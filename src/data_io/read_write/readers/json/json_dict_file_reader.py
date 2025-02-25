import json
from pathlib import Path
from typing import Any, Dict

from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.read_write.readers.file_reader import FileReader


class JSONDictFileReader(FileReader):
    """
    JSON dictionary file reader.
    """

    def read(self, path: Path) -> JSONDictFile:
        """Reads and parses data from a JSON file. Note that the reader assumes
        the outermost structure of the file to be a dictionary and will
        validate the structure against this assumption.

        Args:
            path (Path): The path to the JSON file.

        Returns:
            Any: The parsed data (could be a dict, list, etc.).
        """
        if not path.suffix == ".json":
            raise ValueError(f"Expected a .json file, but got {path.suffix}.")

        if not path.exists():
            raise FileNotFoundError(f"The file at {path} does not exist.")

        with path.open("r", encoding="utf-8") as file:
            json_file = json.load(file)
            if self.__json_file_is_dictionary(json_file):
                return JSONDictFile(data=json_file)
            else:
                raise ValueError(f"JSON file must be dictionary")

    def __json_file_is_dictionary(self, json_file) -> bool:
        return isinstance(json_file, Dict)
