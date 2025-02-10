from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.data_model.features.aggregate.descriptive_statistic import DescriptiveStatistic

from data_types.feature.raw_feature_type import RawFeatureType
from data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)


@dataclass
class AggregateFeature:
    """
    Represents aggregate feature of all epochs of a given raw feature set entry
    """

    descriptive_statistics: List[DescriptiveStatistic]
    feature_type: RawFeatureType
    _statistics_map: Dict[DescriptiveStatisticType, DescriptiveStatistic] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        self._statistics_map = {
            statistics.statistic_type: statistics
            for statistics in self.descriptive_statistics
        }

    def get_statistic_from_type(
        self, statistic_type: DescriptiveStatisticType
    ) -> Optional[DescriptiveStatistic]:
        return self._statistics_map.get(statistic_type)
