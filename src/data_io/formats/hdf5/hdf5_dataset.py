from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.data_io.formats.file_format import FileFormat


@dataclass
class HDF5Dataset(FileFormat):
    """HDF5 dataset containing data and attributes."""

    name: str = ""
    data: List[Any] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
