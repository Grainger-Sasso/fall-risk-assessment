from enum import Enum
from typing import Any, Callable

import numpy as np


class DescriptiveStatisticType(Enum):
    """
    Collection of descriptive statistic types
    """

    PLACEHOLDER = "placeholder"
    # mean, median, std, min, max, percentiles (25th, 75th), coeff of variation, inter-quartile range
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    PERCENTILE_25 = "25th_percentile"
    PERCENTILE_75 = "75th_percentile"
    COEFF_VARIATION = "coefficient_of_variation"
    IQR = "interquartile_range"

    @classmethod
    def get_function(
        cls, stat_type: "DescriptiveStatisticType"
    ) -> Callable[[Any], float]:
        """Returns the corresponding numpy function for a given statistic type"""
        function_map = {
            cls.MEAN: np.nanmean,
            cls.MEDIAN: np.nanmedian,
            cls.STD: np.nanstd,
            cls.MIN: np.nanmin,
            cls.MAX: np.nanmax,
            cls.PERCENTILE_25: lambda x: np.nanpercentile(x, 25),
            cls.PERCENTILE_75: lambda x: np.nanpercentile(x, 75),
            cls.COEFF_VARIATION: lambda x: (
                np.nanstd(x) / np.nanmean(x) if np.nanmean(x) != 0 else np.nan
            ),
            cls.IQR: lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25),
        }
        return function_map[stat_type]

    def __call__(self, data: Any) -> float:
        """Allows direct calling of the enum member to compute the statistic"""
        return self.get_function(self)(data)
