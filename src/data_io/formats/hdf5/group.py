from dataclasses import dataclass
from typing import Any, List, Dict, Union

from src.data_io.formats.hdf5.dataset import Dataset


@dataclass
class Group:
    name: str
    items: List[Union["Group", Dataset]]
    attributes: Dict[str, Any]
