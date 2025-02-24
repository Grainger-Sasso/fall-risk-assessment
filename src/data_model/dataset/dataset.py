from dataclasses import dataclass
from typing import List

from src.data_model.dataset.dataset_entry import DatasetEntry


@dataclass
class Dataset:
    name: str
    entries: List[DatasetEntry]
