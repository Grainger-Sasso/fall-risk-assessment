from dataclasses import dataclass
from typing import Dict, Any

from src.data_io.formats.file_format import FileFormat


@dataclass
class JSONDictFile(FileFormat):
    """
    Represents a JSON file.
    """

    data: Dict[str, Any]  # Data from JSON file
