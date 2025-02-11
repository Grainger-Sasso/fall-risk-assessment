from dataclasses import dataclass
from typing import Any, List, Dict


@dataclass
class Dataset:
    name: str
    data: List[Any]
    attributes: Dict[str, Any]
