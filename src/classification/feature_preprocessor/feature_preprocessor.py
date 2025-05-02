from typing import Dict, List, Union

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.data_model.data.user.user_data import UserData
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier


class FeatureGroup:
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features: np.ndarray = features
        self.labels: np.ndarray = labels


class TrainingData:
    def __init__(
        self, feature_names, test_group: FeatureGroup, train_group: FeatureGroup
    ):
        self.feature_names = feature_names
        self.test_group: FeatureGroup = test_group
        self.train_group: FeatureGroup = train_group


class FeaturePreprocessor:
    def __init__(
        self, feature_ids: List[AggregateFeatureIdentifier], db_manager: DatabaseManager
    ):
        self.feature_ids: List[AggregateFeatureIdentifier] = feature_ids
        self.db_manager: DatabaseManager = db_manager

    def preprocess(self):
        features: List[List[float]] = []
        labels: List[int] = []
        feature_names = []
        for feature_id in self.feature_ids:
            agg_feature_set: AggregateFeatureSetEntry = self.db_manager.import_data(
                [feature_id]
            )
            user_id: UserIdentifier = agg_feature_set.metadata.user_identifier
            user_data: UserData = self.db_manager.import_data([user_id])
            features.append(self.collect_features(agg_feature_set))
            labels.append(user_data.clinical_demographic_data.faller_status.to_bool())
            feature_names = (
                self.get_feauture_names(agg_feature_set)
                if not feature_names
                else feature_names
            )
        X = np.array(features)
        y = np.array(labels)
        # Impute data for missing values
        imputer = SimpleImputer(strategy="median")  # or 'mean', 'most_frequent'
        X_imputed = imputer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed,
            y,
            test_size=0.2,  # 20% for testing
            stratify=y,  # Critical for imbalanced classes
            random_state=42,  # Reproducibility
        )
        return TrainingData(
            feature_names=feature_names,
            test_group=FeatureGroup(X_test, y_test),
            train_group=FeatureGroup(X_train, y_train),
        )

    def collect_features(
        self, agg_feature_set: AggregateFeatureSetEntry
    ) -> List[float]:
        features = []
        for agg_feature in agg_feature_set.aggregate_features:
            for stat in agg_feature.descriptive_statistics:
                features.append[stat.value]
        return features

    def get_feauture_names(self, agg_feature_set: AggregateFeatureSetEntry):
        names = []
        for agg_feature in agg_feature_set.aggregate_features:
            feature_type: RawFeatureType = agg_feature.feature_type
            for stat in agg_feature.descriptive_statistics:
                stat_type: DescriptiveStatisticType = stat.statistic_type
                names.append((feature_type, stat_type))
        return names
