from dataclasses import dataclass
from typing import List

from src.data_model.dataset.entry import Entry


@dataclass
class Dataset:
    name: str
    entries = List[Entry]
