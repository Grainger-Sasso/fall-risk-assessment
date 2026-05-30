from typing import Any, Dict, List, Tuple

import numpy as np
from src.data_model.features.aggregate.aggregate_feature import AggregateFeature
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.aggregate.descriptive_statistic import DescriptiveStatistic
from src.data_model.features.aggregate.metadata.aggregate_feature_set_entry_metadata import (
    AggregateFeatureSetEntryMetadata,
)
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_epoch_features import RawEpochFeature
from src.data_model.features.raw.raw_feature import RawFeature
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from data_types.feature.feature_type import FeatureType
from src.gait_features.feature_extraction.gait_feature_extractor import GaitResults
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.identifier import IdentifierGenerator


class GaitFeatureProcessor:
    def __init__(self):
        self.raw_feat_id_gen = IdentifierGenerator("raw", RawFeatureIdentifier)
        self.agg_feat_id_gen = IdentifierGenerator("agg", AggregateFeatureIdentifier)
        self.event_gait_features: List[FeatureType] = [
            FeatureType.STRIDE_TIME,
            FeatureType.STRIDE_TIME_ASYMMETRY,
            FeatureType.STANCE_TIME,
            FeatureType.STANCE_TIME_ASYMMETRY,
            FeatureType.SWING_TIME,
            FeatureType.SWING_TIME_ASYMMETRY,
            FeatureType.STEP_TIME,
            FeatureType.STEP_TIME_ASYMMETRY,
            FeatureType.INITIAL_DOUBLE_SUPPORT,
            FeatureType.INITIAL_DOUBLE_SUPPORT_ASYMMETRY,
            FeatureType.TERMINAL_DOUBLE_SUPPORT,
            FeatureType.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY,
            FeatureType.DOUBLE_SUPPORT,
            FeatureType.DOUBLE_SUPPORT_ASYMMETRY,
            FeatureType.SINGLE_SUPPORT,
            FeatureType.SINGLE_SUPPORT_ASYMMETRY,
            FeatureType.M2_DELTA_H,
            FeatureType.M2_DELTA_H_PRIME,
            FeatureType.STEP_LENGTH,
            FeatureType.STEP_LENGTH_ASYMMETRY,
            FeatureType.STRIDE_LENGTH,
            FeatureType.STRIDE_LENGTH_ASYMMETRY,
            FeatureType.GAIT_SPEED,
            FeatureType.GAIT_SPEED_ASYMMETRY,
            FeatureType.CADENCE,
            FeatureType.M1_DELTA_H,
            FeatureType.STEP_LENGTH_M1,
            FeatureType.STEP_LENGTH_M1_ASYMMETRY,
            FeatureType.STRIDE_LENGTH_M1,
            FeatureType.STRIDE_LENGTH_M1_ASYMMETRY,
            FeatureType.GAIT_SPEED_M1,
            FeatureType.GAIT_SPEED_M1_ASYMMETRY,
            FeatureType.INTRA_STEP_COVARIANCE_V,
            FeatureType.INTRA_STRIDE_COVARIANCE_V,
            FeatureType.HARMONIC_RATIO_V,
            FeatureType.STRIDE_SPARC,
        ]
        self.bout_gait_features: List[FeatureType] = [
            FeatureType.BOUT_DURATION,
            FeatureType.BOUT_STEPS,
            FeatureType.GAIT_CYCLES,
            FeatureType.DEBUG_MEAN_STEP_FREQ,
            FeatureType.BOUT_PHASE_COORDINATION_INDEX,
            FeatureType.BOUT_GAIT_SYMMETRY_INDEX,
            FeatureType.BOUT_STEP_REGULARITY_V,
            FeatureType.BOUT_STRIDE_REGULARITY_V,
            FeatureType.BOUT_AUTOCOVARIANCE_SYMMETRY_V,
            FeatureType.BOUT_REGULARITY_INDEX_V,
        ]
        self.raw_feature_types: List[FeatureType] = [
            FeatureType.STRIDE_TIME,
            FeatureType.STRIDE_TIME_ASYMMETRY,
            FeatureType.STANCE_TIME,
            FeatureType.STANCE_TIME_ASYMMETRY,
            FeatureType.SWING_TIME,
            FeatureType.SWING_TIME_ASYMMETRY,
            FeatureType.STEP_TIME,
            FeatureType.STEP_TIME_ASYMMETRY,
            FeatureType.INITIAL_DOUBLE_SUPPORT,
            FeatureType.INITIAL_DOUBLE_SUPPORT_ASYMMETRY,
            FeatureType.TERMINAL_DOUBLE_SUPPORT,
            FeatureType.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY,
            FeatureType.DOUBLE_SUPPORT,
            FeatureType.DOUBLE_SUPPORT_ASYMMETRY,
            FeatureType.SINGLE_SUPPORT,
            FeatureType.SINGLE_SUPPORT_ASYMMETRY,
            FeatureType.M2_DELTA_H,
            FeatureType.M2_DELTA_H_PRIME,
            FeatureType.STEP_LENGTH,
            FeatureType.STEP_LENGTH_ASYMMETRY,
            FeatureType.STRIDE_LENGTH,
            FeatureType.STRIDE_LENGTH_ASYMMETRY,
            FeatureType.GAIT_SPEED,
            FeatureType.GAIT_SPEED_ASYMMETRY,
            FeatureType.CADENCE,
            FeatureType.M1_DELTA_H,
            FeatureType.STEP_LENGTH_M1,
            FeatureType.STEP_LENGTH_M1_ASYMMETRY,
            FeatureType.STRIDE_LENGTH_M1,
            FeatureType.STRIDE_LENGTH_M1_ASYMMETRY,
            FeatureType.GAIT_SPEED_M1,
            FeatureType.GAIT_SPEED_M1_ASYMMETRY,
            FeatureType.INTRA_STEP_COVARIANCE_V,
            FeatureType.INTRA_STRIDE_COVARIANCE_V,
            FeatureType.HARMONIC_RATIO_V,
            FeatureType.STRIDE_SPARC,
            FeatureType.BOUT_DURATION,
            FeatureType.BOUT_STEPS,
            FeatureType.GAIT_CYCLES,
            FeatureType.DEBUG_MEAN_STEP_FREQ,
            FeatureType.BOUT_PHASE_COORDINATION_INDEX,
            FeatureType.BOUT_GAIT_SYMMETRY_INDEX,
            FeatureType.BOUT_STEP_REGULARITY_V,
            FeatureType.BOUT_STRIDE_REGULARITY_V,
            FeatureType.BOUT_AUTOCOVARIANCE_SYMMETRY_V,
            FeatureType.BOUT_REGULARITY_INDEX_V,
        ]

    def process_features(
        self, gait_res: GaitResults, imu_data: IMUData, user_data: UserData
    ) -> Tuple[RawFeatureSetEntry, AggregateFeatureSetEntry]:
        multiday_features: List[Dict[FeatureType, float]] = (
            self._aggregate_multiday_features(gait_res)
        )
        agg_feature_dict: Dict[FeatureType, Dict[DescriptiveStatisticType, float]] = (
            self._compute_aggregate_features(multiday_features)
        )
        raw_feature_set_entry = self._convert_multiday_to_raw_features(
            multiday_features, imu_data, user_data
        )
        agg_feature_set_entry = self._convert_agg_features_to_data_model(
            agg_feature_dict,
            imu_data,
            user_data,
            raw_feature_set_entry.metadata.raw_feature_identifier,
        )
        return raw_feature_set_entry, agg_feature_set_entry

    def _convert_multiday_to_raw_features(
        self,
        multiday_features: List[Dict[FeatureType, float]],
        imu_data: IMUData,
        user_data: UserData,
    ) -> RawFeatureSetEntry:
        # Build list of epoch features
        epoch_features: List[RawEpochFeature] = []
        for features in multiday_features:
            raw_features: List[RawFeature] = []
            for raw_feature_type in self.raw_feature_types:
                raw_features.append(
                    RawFeature(raw_feature_type, features[raw_feature_type])
                )
            epoch_features.append(
                RawEpochFeature(
                    raw_features,
                    features[FeatureType.BOUT_START_TIMESTAMP],
                    features[FeatureType.BOUT_END_TIMESTAMP],
                )
            )
        # Build raw feature metadata
        metadata = RawFeatureSetEntryMetadata(
            raw_feature_identifier=self.raw_feat_id_gen.generate_identifier(),
            user_identifier=user_data.user_identifier,
            imu_data_identifier=imu_data.get_data_id(),
            start_time=epoch_features[0].epoch_start_time,
            # DUMMY VALUE, TO REMOVE
            epoch_length=1.0,
        )
        return RawFeatureSetEntry(raw_epoch_features=epoch_features, metadata=metadata)

    def _convert_agg_features_to_data_model(
        self,
        agg_feature_dict: Dict[FeatureType, Dict[DescriptiveStatisticType, float]],
        imu_data: IMUData,
        user_data: UserData,
        raw_feature_id: RawFeatureIdentifier,
    ) -> AggregateFeatureSetEntry:
        # Build agg feature list
        agg_feature_list: List[AggregateFeature] = []
        for feature_type, stats in agg_feature_dict.items():
            stat_list = []
            for stat_type, stat_value in stats.items():
                stat_list.append(DescriptiveStatistic(stat_type, stat_value))
            agg_feature_list.append((AggregateFeature(stat_list, feature_type)))
        # Build agg feature metadata
        metadata = AggregateFeatureSetEntryMetadata(
            aggregate_feature_identifier=self.agg_feat_id_gen.generate_identifier(),
            raw_feature_identifier=raw_feature_id,
            user_identifier=user_data.user_identifier,
            imu_data_identifier=imu_data.get_data_id(),
        )
        return AggregateFeatureSetEntry(
            aggregate_features=agg_feature_list, metadata=metadata
        )

    def _aggregate_multiday_features(
        self, gait_res: GaitResults
    ) -> List[Dict[FeatureType, float]]:
        """Aggregates multiday, event-level features into collection of bout-level features

        Args:
            gait_res (Dict): _description_

        Returns:
            List[Dict[GaitFeatureKeys, float]]: _description_
        """
        # Init results dictionary (dayN - boutN)
        multiday_features: List[Dict[FeatureType, float]] = []
        # Initialize pointers for to traverse days and bouts
        day_start_ix = 0
        day_n = 1
        # Reference bout numbers and days from resutls
        day_n_list = gait_res.data[FeatureType.DAY_N.value]
        # Traverse days
        while day_start_ix < len(day_n_list):
            day_end_ix = day_start_ix
            # Traverse days to find end day index
            while day_end_ix < len(day_n_list) and day_n_list[day_end_ix] == day_n:
                day_end_ix += 1
            # Aggregate bouts, add to result
            bout_features = self._aggregate_single_day_features(
                gait_res.data, day_start_ix, day_end_ix
            )
            multiday_features.extend(bout_features)
            # Increment day_n and day_start_ix
            day_n += 1
            day_start_ix = day_end_ix
        # Return results
        return multiday_features

    def _aggregate_single_day_features(
        self, gait_res, day_start_ix: int, day_end_ix: int
    ) -> List[Dict[FeatureType, np.float64]]:
        single_day_features = []
        bout_n_list = gait_res[FeatureType.BOUT_N.value]
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
            bout_features[FeatureType.BOUT_START_TIMESTAMP] = gait_res[
                FeatureType.IC_TIME.value
            ][bout_start_ix - 1].timestamp()
            bout_features[FeatureType.BOUT_END_TIMESTAMP] = gait_res[
                FeatureType.IC_TIME.value
            ][bout_end_ix - 1].timestamp()
            single_day_features.append(bout_features)
            bout_n += 1
            bout_start_ix = bout_end_ix
        return single_day_features

    def _compute_aggregate_features(
        self,
        multiday_features: List[Dict[FeatureType, float]],
    ) -> Dict[FeatureType, Dict[DescriptiveStatisticType, float]]:
        aggregate_features: Dict[FeatureType, Dict[DescriptiveStatisticType, float]] = (
            {}
        )
        for raw_feature_type in self.raw_feature_types:
            feature_values = np.array(
                [features[raw_feature_type] for features in multiday_features]
            )
            descriptive_stats: Dict[DescriptiveStatisticType, float] = {}
            for type in DescriptiveStatisticType:
                descriptive_stats[type] = type(feature_values)
            aggregate_features[raw_feature_type] = descriptive_stats
        return aggregate_features
