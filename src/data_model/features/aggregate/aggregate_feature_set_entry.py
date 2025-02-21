from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.data_types.feature.raw_feature_type import RawFeatureType
from src.data_model.features.aggregate.aggregate_feature import AggregateFeature
from src.data_model.features.aggregate.metadata.aggregate_feature_set_entry_metadata import (
    AggregateFeatureSetEntryMetadata,
)


@dataclass
class AggregateFeatureSetEntry:
    """
    Represents aggregate features of feature types of a given raw feature set entry
    """

    aggregate_features: List[AggregateFeature]
    metadata: AggregateFeatureSetEntryMetadata
    _feature_map: Dict[RawFeatureType, AggregateFeature] = field(init=False, repr=False)

    def __post_init__(self):
        self._feature_map = {
            feature.feature_type: feature for feature in self.aggregate_features
        }

    def get_feature_from_type(
        self, feature_type: RawFeatureType
    ) -> Optional[AggregateFeature]:
        """
        Retrieve an AggregateFeature by its RawFeatureType from the _feature_map.

        Args:
            feature_type (RawFeatureType): The feature type to look up.

        Returns:
            Optional[AggregateFeature]: The corresponding AggregateFeature, or None if not found.
        """
        return self._feature_map.get(feature_type)
