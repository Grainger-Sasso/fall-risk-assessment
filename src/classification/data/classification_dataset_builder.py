from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.classification.data.classification_dataset import (
    BasisParticipantDataset,
    BasisSampleDataset,
    ClassificationDataset,
    EarlyFusionParticipantDataset,
)
from src.classification.data.participant_aggregation import (
    aggregate_samples_to_participants,
)
from src.data_model.features.bout_sample_layout import (
    flatten_bout_features,
    usable_sample_mask,
)
from src.data_types.feature.feature_type import FeatureType
from src.data_types.feature.stride_feature_name import StrideFeatureName
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

    def build(self, stride_catalog: Optional[str] = None) -> ClassificationDataset:
        return ClassificationDataset(
            stride=self.build_basis(SampleBasis.STRIDE, stride_catalog=stride_catalog),
            epoch=self.build_basis(SampleBasis.EPOCH),
        )

    def build_participant_basis(
        self,
        basis: SampleBasis,
        aggregation: str = "mean",
        participant_order: Optional[List[str]] = None,
        stride_catalog: Optional[str] = None,
    ) -> BasisParticipantDataset:
        sample_dataset = self.build_basis(basis, stride_catalog=stride_catalog)
        order = participant_order or sample_dataset.participant_ids()
        x_rows, labels, kept_ids = aggregate_samples_to_participants(
            sample_dataset,
            participant_order=order,
            aggregation=aggregation,
        )
        return BasisParticipantDataset(
            basis=basis,
            X=x_rows,
            y=labels,
            participant_ids=kept_ids,
            feature_names=list(sample_dataset.feature_names),
            aggregation=aggregation,
        )

    def build_early_fusion_participant_dataset(
        self,
        aggregation: str = "mean",
        stride_catalog: Optional[str] = None,
    ) -> EarlyFusionParticipantDataset:
        sample_dataset = self.build(stride_catalog=stride_catalog)
        common_ids = sample_dataset.common_participant_ids()
        stride = self.build_participant_basis(
            SampleBasis.STRIDE,
            aggregation=aggregation,
            participant_order=common_ids,
            stride_catalog=stride_catalog,
        )
        epoch = self.build_participant_basis(
            SampleBasis.EPOCH,
            aggregation=aggregation,
            participant_order=common_ids,
        )
        if stride.is_empty or epoch.is_empty:
            return EarlyFusionParticipantDataset(
                X=np.empty((0, 0)),
                y=np.empty((0,), dtype=int),
                participant_ids=[],
                stride_feature_names=stride.feature_names,
                epoch_feature_names=epoch.feature_names,
                aggregation=aggregation,
            )

        if stride.participant_ids != epoch.participant_ids:
            raise ValueError(
                "Stride and epoch participant aggregation produced mismatched "
                "participant order."
            )

        return EarlyFusionParticipantDataset(
            X=np.hstack([stride.X, epoch.X]),
            y=stride.y,
            participant_ids=list(stride.participant_ids),
            stride_feature_names=list(stride.feature_names),
            epoch_feature_names=list(epoch.feature_names),
            aggregation=aggregation,
        )

    def build_basis(
        self,
        basis: SampleBasis,
        stride_catalog: Optional[str] = None,
    ) -> BasisSampleDataset:
        reference_types: Optional[List[StrideFeatureName]] = None
        reference_catalog: Optional[str] = None
        x_blocks: List[np.ndarray] = []
        y_values: List[int] = []
        group_values: List[str] = []

        for feature_id in self.db_manager.list_feature_ids():
            record = self.db_manager.load_features(feature_id)
            metadata = record.feature_metadata
            if basis == SampleBasis.STRIDE:
                record_catalog = metadata.stride_feature_catalog
                if stride_catalog is not None and record_catalog != stride_catalog:
                    continue
                if reference_catalog is None:
                    reference_catalog = record_catalog
                elif record_catalog != reference_catalog:
                    raise ValueError(
                        "Mixed stride feature catalogs detected in feature store: "
                        f"'{reference_catalog}' and '{record_catalog}'. "
                        "Rebuild features with a single extraction backend or pass "
                        "stride_catalog to filter records."
                    )

            bout = record.get_features_by_basis(basis)
            record_types = list(bout.feature_names)
            if reference_types is None:
                reference_types = record_types

            features = np.asarray(bout.features, dtype=float)
            if features.size == 0:
                continue

            per_sample = flatten_bout_features(features)
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

            aligned = aligned[usable_sample_mask(aligned)]
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
