from dataclasses import dataclass
from typing import List

from data_model.dataset.dataset_entry import Entry


@dataclass
class Dataset:
    name: str
    entries: List[Entry]
