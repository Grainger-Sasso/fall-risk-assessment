from dataclasses import dataclass
from typing import List

from src.data_model.feature_set.feature_set_entry import FeatureSetEntry


@dataclass
class FeatureSet:
    name: str
    entries: List[FeatureSetEntry]
