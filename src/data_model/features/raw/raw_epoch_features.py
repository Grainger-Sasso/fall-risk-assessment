from dataclasses import dataclass, field
from typing import List, Dict, Optional

from data_types.feature.raw_feature_type import RawFeatureType
from src.data_model.features.raw.raw_feature import RawFeature


@dataclass
class RawEpochFeatures:
    """
    Represents all raw unique features for given epoch
    """

    raw_features: List[RawFeature]
    epoch_start_time: float
    epoch_end_time: float
    _raw_feature_map: Dict[RawFeatureType, RawFeature] = field(init=False, repr=False)

    def __post_init__(self):
        self._raw_feature_map = {
            raw_features.feature_type: raw_features
            for raw_features in self.raw_features
        }

    @property
    def epoch_timestamps(self) -> List[float]:
        """
        Returns the start and end time as a list.
        """
        return [self.epoch_start_time, self.epoch_end_time]

    def get_feature_from_type(
        self, feature_type: RawFeatureType
    ) -> Optional[RawFeature]:
        """
        Retrieve a RawFeature by its RawFeatureType from the _feature_map.

        Args:
            feature_type (RawFeatureType): The feature type to look up.

        Returns:
            Optional[AggregateFeature]: The corresponding AggregateFeature, or None if not found.
        """
        return self._raw_feature_map.get(feature_type)
