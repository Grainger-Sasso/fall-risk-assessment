from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.classification.data.classification_dataset import (
    BasisSampleDataset,
    ClassificationDataset,
)
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.database_manager.database_manager import DatabaseManager


@dataclass
class ClassificationDatasetBuilder:
    """
    Build sample-level classification datasets from the SQLite-backed
    DatabaseManager, one per sampling basis.

    The per-sample flatten/align logic mirrors
    ``VisualizationDataService.collect_population_feature_matrix`` but is
    specialized for supervised learning: rows inherit the participant's binary
    faller label, ``unknown`` participants are dropped, and all-NaN sample rows
    are removed.
    """

    db_manager: DatabaseManager

    def build(self) -> ClassificationDataset:
        return ClassificationDataset(
            stride=self.build_basis(SampleBasis.STRIDE),
            epoch=self.build_basis(SampleBasis.EPOCH),
        )

    def build_basis(self, basis: SampleBasis) -> BasisSampleDataset:
        reference_types: Optional[List[FeatureType]] = None
        x_blocks: List[np.ndarray] = []
        y_values: List[int] = []
        group_values: List[str] = []

        for feature_id in self.db_manager.list_feature_ids():
            record = self.db_manager.load_features(feature_id)
            bout = record.get_features_by_basis(basis)
            record_types = list(bout.feature_names)
            if reference_types is None:
                reference_types = record_types

            features = np.asarray(bout.features, dtype=float)
            if features.size == 0:
                continue

            num_bouts, _, num_samples = features.shape
            per_sample = np.transpose(features, (0, 2, 1)).reshape(
                num_bouts * num_samples, len(record_types)
            )

            aligned = np.full((per_sample.shape[0], len(reference_types)), np.nan)
            index_by_type = {ftype: idx for idx, ftype in enumerate(record_types)}
            for col, ftype in enumerate(reference_types):
                source_index = index_by_type.get(ftype)
                if source_index is not None:
                    aligned[:, col] = per_sample[:, source_index]

            user_id = record.feature_metadata.user_identifier
            user = self.db_manager.load_user(user_id)
            label = user.clinical_demographic_data.faller_status.to_bool()
            if label is None:
                continue

            valid_rows = ~np.all(np.isnan(aligned), axis=1)
            aligned = aligned[valid_rows]
            if aligned.shape[0] == 0:
                continue

            x_blocks.append(aligned)
            y_values.extend([int(label)] * aligned.shape[0])
            group_values.extend([user_id.value] * aligned.shape[0])

        if reference_types is None or not x_blocks:
            return BasisSampleDataset(
                basis=basis,
                X=np.empty((0, 0)),
                y=np.empty((0,), dtype=int),
                groups=np.empty((0,), dtype=object),
                feature_names=reference_types or [],
            )

        return BasisSampleDataset(
            basis=basis,
            X=np.vstack(x_blocks),
            y=np.asarray(y_values, dtype=int),
            groups=np.asarray(group_values, dtype=object),
            feature_names=reference_types,
        )
