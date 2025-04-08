from typing import Any, Dict, List

import numpy as np

from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier


class FeatureIDGenerator:
    def generate_raw_feature_id(self):
        return RawFeatureIdentifier("1234")

    def generate_agg_feature_id(self):
        return AggregateFeatureIdentifier("5678")


class GaitFeatureProcessor:
    def __init__(self):
        self.event_gait_features: List[RawFeatureType] = [
            RawFeatureType.STRIDE_TIME,
            RawFeatureType.STRIDE_TIME_ASYMMETRY,
            RawFeatureType.STANCE_TIME,
            RawFeatureType.STANCE_TIME_ASYMMETRY,
            RawFeatureType.SWING_TIME,
            RawFeatureType.SWING_TIME_ASYMMETRY,
            RawFeatureType.STEP_TIME,
            RawFeatureType.STEP_TIME_ASYMMETRY,
            RawFeatureType.INITIAL_DOUBLE_SUPPORT,
            RawFeatureType.INITIAL_DOUBLE_SUPPORT_ASYMMETRY,
            RawFeatureType.TERMINAL_DOUBLE_SUPPORT,
            RawFeatureType.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY,
            RawFeatureType.DOUBLE_SUPPORT,
            RawFeatureType.DOUBLE_SUPPORT_ASYMMETRY,
            RawFeatureType.SINGLE_SUPPORT,
            RawFeatureType.SINGLE_SUPPORT_ASYMMETRY,
            RawFeatureType.M2_DELTA_H,
            RawFeatureType.M2_DELTA_H_PRIME,
            RawFeatureType.STEP_LENGTH,
            RawFeatureType.STEP_LENGTH_ASYMMETRY,
            RawFeatureType.STRIDE_LENGTH,
            RawFeatureType.STRIDE_LENGTH_ASYMMETRY,
            RawFeatureType.GAIT_SPEED,
            RawFeatureType.GAIT_SPEED_ASYMMETRY,
            RawFeatureType.CADENCE,
            RawFeatureType.M1_DELTA_H,
            RawFeatureType.STEP_LENGTH_M1,
            RawFeatureType.STEP_LENGTH_M1_ASYMMETRY,
            RawFeatureType.STRIDE_LENGTH_M1,
            RawFeatureType.STRIDE_LENGTH_M1_ASYMMETRY,
            RawFeatureType.GAIT_SPEED_M1,
            RawFeatureType.GAIT_SPEED_M1_ASYMMETRY,
            RawFeatureType.INTRA_STEP_COVARIANCE_V,
            RawFeatureType.INTRA_STRIDE_COVARIANCE_V,
            RawFeatureType.HARMONIC_RATIO_V,
            RawFeatureType.STRIDE_SPARC,
        ]
        self.bout_gait_features: List[RawFeatureType] = [
            RawFeatureType.BOUT_DURATION,
            RawFeatureType.BOUT_STEPS,
            RawFeatureType.GAIT_CYCLES,
            RawFeatureType.DEBUG_MEAN_STEP_FREQ,
            RawFeatureType.BOUT_PHASE_COORDINATION_INDEX,
            RawFeatureType.BOUT_GAIT_SYMMETRY_INDEX,
            RawFeatureType.BOUT_STEP_REGULARITY_V,
            RawFeatureType.BOUT_STRIDE_REGULARITY_V,
            RawFeatureType.BOUT_AUTOCOVARIANCE_SYMMETRY_V,
            RawFeatureType.BOUT_REGULARITY_INDEX_V,
        ]

    def process_features(self, gait_res: Dict[str, Any]):
        multiday_features: List[Dict[RawFeatureType, float]] = (
            self._aggregate_multiday_features(gait_res)
        )
        raw_feature_types: List[RawFeatureType] = multiday_features[0].keys()
        agg_features: Dict[RawFeatureType, Dict[DescriptiveStatisticType, float]] = (
            self._compute_aggregate_features(multiday_features, raw_feature_types)
        )

    def _convert_multiday_to_raw_features(
        self, multiday_features: List[Dict[RawFeatureType, float]]
    ):
        pass

    def _convert_agg_features_to_data_model(
        self, agg_features: Dict[RawFeatureType, Dict[DescriptiveStatisticType, float]]
    ):
        pass

    def _aggregate_multiday_features(
        self, gait_res: Dict[str, Any]
    ) -> List[Dict[RawFeatureType, float]]:
        """Aggregates multiday, event-level features into collection of bout-level features

        Args:
            gait_res (Dict): _description_

        Returns:
            List[Dict[GaitFeatureKeys, float]]: _description_
        """
        # Init results dictionary (dayN - boutN)
        multiday_features: List[Dict[RawFeatureType, float]] = []
        # Initialize pointers for to traverse days and bouts
        day_start_ix = 0
        day_n = 1
        # Reference bout numbers and days from resutls
        day_n_list = gait_res[RawFeatureType.DAY_N.value]
        # Traverse days
        while day_start_ix < len(day_n_list):
            day_end_ix = day_start_ix
            # Traverse days to find end day index
            while day_end_ix < len(day_n_list) and day_n_list[day_end_ix] == day_n:
                day_end_ix += 1
            # Aggregate bouts, add to result
            bout_features = self._aggregate_single_day_features(
                gait_res, day_start_ix, day_end_ix
            )
            multiday_features.extend(bout_features)
            # Increment day_n and day_start_ix
            day_n += 1
            day_start_ix = day_end_ix
        # Return results
        return multiday_features

    def _aggregate_single_day_features(
        self, gait_res, day_start_ix: int, day_end_ix: int
    ) -> List[Dict[RawFeatureType, np.float64]]:
        single_day_features = []
        bout_n_list = gait_res[RawFeatureType.BOUT_N.value]
        bout_start_ix = day_start_ix
        bout_n = 1
        while bout_start_ix < len(bout_n_list) and bout_start_ix < day_end_ix:
            bout_features = {}
            bout_end_ix = bout_start_ix
            while bout_end_ix < day_end_ix and bout_n_list[bout_end_ix] == bout_n:
                bout_end_ix += 1

            for event_metric in self.event_gait_features:
                # Take mean of features ignoring nan values
                bout_features[event_metric] = np.nanmean(
                    gait_res[event_metric.value][bout_start_ix:bout_end_ix]
                )
            for bout_metric in self.bout_gait_features:
                bout_features[bout_metric] = np.float64(
                    gait_res[bout_metric.value][bout_start_ix]
                )
            single_day_features.append(bout_features)
            bout_n += 1
            bout_start_ix = bout_end_ix
        return single_day_features

    def _compute_aggregate_features(
        self,
        multiday_features: List[Dict[RawFeatureType, float]],
        raw_feature_types: List[RawFeatureType],
    ) -> Dict[RawFeatureType, Dict[DescriptiveStatisticType, float]]:
        aggregate_features: Dict[
            RawFeatureType, Dict[DescriptiveStatisticType, float]
        ] = {}
        for raw_feature_type in raw_feature_types:
            feature_values = np.array(
                [features[raw_feature_type] for features in multiday_features]
            )
            descriptive_stats: Dict[DescriptiveStatisticType, float] = {}
            for type in DescriptiveStatisticType:
                descriptive_stats[type] = type(feature_values)
            aggregate_features[raw_feature_type] = aggregate_features
        return aggregate_features
