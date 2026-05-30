from typing import List, Optional, Tuple

import numpy as np
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)

from src.data_model.data.user.user_data import UserData
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from data_types.feature.feature_type import FeatureType
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier


class PreparedDataset:
    def __init__(
        self,
        feature_names,
        features: np.ndarray,
        labels: np.ndarray,
        user_ids: List[str],
        aggregate_feature_ids: List[str],
    ):
        self.feature_names = feature_names
        self.features: np.ndarray = features
        self.labels: np.ndarray = labels
        self.user_ids: List[str] = user_ids
        self.aggregate_feature_ids: List[str] = aggregate_feature_ids


class FeaturePreprocessor:
    def __init__(
        self,
        feature_ids: List[AggregateFeatureIdentifier],
        db_manager: DatabaseManager,
        selected_feature_pairs: Optional[
            List[Tuple[FeatureType, DescriptiveStatisticType]]
        ] = None,
    ):
        self.feature_ids: List[AggregateFeatureIdentifier] = feature_ids
        self.db_manager: DatabaseManager = db_manager
        self.selected_feature_pairs = selected_feature_pairs

    def preprocess(self) -> PreparedDataset:
        """
        Build full participant-level feature matrix and label vector.

        Preprocessing transformations (imputation/scaling) should be done inside
        cross-validation folds by the classifier pipeline to avoid leakage.
        """
        features: List[List[float]] = []
        labels: List[int] = []
        user_ids: List[str] = []
        aggregate_feature_ids: List[str] = []
        feature_names: List[Tuple[FeatureType, DescriptiveStatisticType]] = []

        for feature_id in self.feature_ids:
            agg_feature_set_entry: AggregateFeatureSetEntry = (
                self.db_manager.import_data([feature_id])[0]
            )
            user_id: UserIdentifier = agg_feature_set_entry.metadata.user_identifier
            user_data: UserData = self.db_manager.import_data([user_id])[0]
            label_value = user_data.clinical_demographic_data.faller_status.to_bool()
            # Skip unlabeled users for binary classifier training.
            if label_value is None:
                continue

            if not feature_names:
                if self.selected_feature_pairs is not None:
                    feature_names = self.selected_feature_pairs
                else:
                    feature_names = self.get_feature_names(agg_feature_set_entry)
            features.append(self.collect_features(agg_feature_set_entry, feature_names))
            labels.append(int(label_value))
            user_ids.append(user_id.value)
            aggregate_feature_ids.append(feature_id.value)

        if not features:
            raise ValueError(
                "No labeled aggregate feature rows available for preprocessing."
            )

        X = np.array(features, dtype=float)
        y = np.array(labels)
        X, feature_names = self._drop_all_missing_columns(X, feature_names)
        if X.shape[1] == 0:
            raise ValueError(
                "No usable feature columns remain after dropping all-missing columns."
            )
        return PreparedDataset(
            feature_names=feature_names,
            features=X,
            labels=y,
            user_ids=user_ids,
            aggregate_feature_ids=aggregate_feature_ids,
        )

    def collect_features(
        self,
        agg_feature_set: AggregateFeatureSetEntry,
        feature_names: List[Tuple[FeatureType, DescriptiveStatisticType]],
    ) -> List[float]:
        features: List[float] = []
        for feature_type, stat_type in feature_names:
            agg_feature = agg_feature_set.get_feature_from_type(feature_type)
            if agg_feature is None:
                features.append(np.nan)
                continue
            stat = agg_feature.get_statistic_from_type(stat_type)
            if stat is None or stat.value is None:
                value = np.nan
            else:
                value = float(stat.value)
                if np.isnan(value):
                    value = np.nan
            features.append(value)
        return features

    def get_feature_names(
        self, agg_feature_set: AggregateFeatureSetEntry
    ) -> List[Tuple[FeatureType, DescriptiveStatisticType]]:
        names = []
        for agg_feature in agg_feature_set.aggregate_features:
            feature_type: FeatureType = agg_feature.feature_type
            for stat in agg_feature.descriptive_statistics:
                stat_type: DescriptiveStatisticType = stat.statistic_type
                names.append((feature_type, stat_type))
        return names

    def _drop_all_missing_columns(
        self,
        X: np.ndarray,
        feature_names: List[Tuple[FeatureType, DescriptiveStatisticType]],
    ) -> Tuple[np.ndarray, List[Tuple[FeatureType, DescriptiveStatisticType]]]:
        """Remove columns that are entirely NaN across all samples."""
        keep_mask = ~np.all(np.isnan(X), axis=0)
        filtered_X = X[:, keep_mask]
        filtered_feature_names = [
            feature_name for feature_name, keep in zip(feature_names, keep_mask) if keep
        ]
        return filtered_X, filtered_feature_names
