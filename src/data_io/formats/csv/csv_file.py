from dataclasses import dataclass
from typing import List, Dict, Any

from src.data_io.formats.file_format import FileFormat


@dataclass
class CSVFile(FileFormat):
    """
    Represents a CSV file with column headers and data.
    """

    fieldnames: List[str]  # List of column headers
    data: Dict[str, List[Any]]  # Dictionary mapping column headers to column data
