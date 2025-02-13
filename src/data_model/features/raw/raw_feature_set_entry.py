from dataclasses import dataclass
from typing import List

from src.data_model.features.raw.raw_epoch_features import RawEpochFeatures
from data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)


@dataclass
class RawFeatureSetEntry:
    """
    Represents raw features across all epochs
    """

    raw_epoch_features: List[RawEpochFeatures]
    metadata: RawFeatureSetEntryMetadata

    @property
    def timestamps(self) -> List[float]:
        """
        Returns the start and end time as a list.
        """
        return [self.start_time, self.end_time]
