from dataclasses import dataclass

from src.data_model.features.types.descriptive_statistic_type import (
    DescriptiveStatisticType,
)


@dataclass
class DescriptiveStatistic:
    """
    Represents descriptive statistics of aggregate features
    """

    statistic_type: DescriptiveStatisticType
    value: float
