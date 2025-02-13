import json
from pathlib import Path
from typing import Any, Tuple

from src.data_io.read_write.writers.file_writer import FileWriter
from src.data_io.formats.json.json_dict_file import JSONDictFile


class JSONDictFileWriter(FileWriter):
    """
    JSON dictionary file writer.
    """

    def write(self, path: Path, json_file: JSONDictFile, **kwargs) -> Tuple[bool, str]:
        """
        Writes JSON data to a file.

        Args:
            path (Path): File path to write.
            data (DataType): Data to write. Expected to be a dictionary or list of dictionaries.
            **kwargs: Additional parameters for specific file formats (e.g., compression, delimiter).

        Returns:
            Tuple[bool, str]: (success flag, error message)
        """
        try:
            with open(path, mode="w", encoding="utf-8") as file:
                json.dump(json_file.data, file, **kwargs)
            return True, ""
        except Exception as e:
            return False, str(e)
