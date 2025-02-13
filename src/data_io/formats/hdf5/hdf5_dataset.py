from dataclasses import dataclass
from typing import Any, List, Dict

from src.data_io.formats.file_format import FileFormat


@dataclass
class HDF5Dataset(FileFormat):
    name: str
    data: List[Any]
    attributes: Dict[str, Any]
