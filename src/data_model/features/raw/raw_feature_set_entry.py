from dataclasses import dataclass
from typing import List

from src.data_model.assessment_data import AssessmentData
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_epoch_features import RawEpochFeature
from src.identifiers.identifier import Identifier


@dataclass
class RawFeatureSetEntry(AssessmentData):
    """
    Represents raw features across all epochs
    """

    raw_epoch_features: List[RawEpochFeature]
    metadata: RawFeatureSetEntryMetadata

    @property
    def timestamps(self) -> List[float]:
        """
        Returns the start and end time as a list.
        """
        return [self.start_time, self.end_time]

    def get_data_id(self) -> Identifier:
        return self.metadata.raw_feature_identifier

    def get_associated_data_id(self) -> Identifier:
        return self.metadata.imu_data_identifier
