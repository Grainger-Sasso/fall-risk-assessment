from dataclasses import dataclass

from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)


@dataclass
class DescriptiveStatistic:
    """
    Represents descriptive statistics of aggregate features
    """

    statistic_type: DescriptiveStatisticType
    value: float
