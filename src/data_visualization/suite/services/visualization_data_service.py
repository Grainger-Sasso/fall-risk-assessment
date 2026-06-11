from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.bout_sample_layout import (
    flatten_bout_features,
    usable_sample_mask,
)
from src.data_model.data.user.user_data import UserData
from src.data_model.features.record_features import RecordFeatures
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.data_types.feature.feature_type import FeatureType
from src.data_types.feature.stride_feature_name import StrideFeatureName
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


@dataclass
class PerRecordCoverageMatrix:
    """
    Per feature-file valid-sample fractions for a single sampling basis.

    ``matrix[i, j]`` is the fraction of *usable* samples in record *i* that have
    a finite value for feature type *j*. Usable samples exclude structurally
    padded NaN slots in the ``(bout, feature, sample)`` tensor. Rows with no
    usable samples use NaN.
    """

    feature_types: List[FeatureType]
    matrix: np.ndarray
    feature_ids: List[str]
    participant_ids: List[str]
    class_labels: List[str]
    usable_sample_counts: List[int]

    @property
    def is_empty(self) -> bool:
        return self.matrix.size == 0 or not self.feature_types

    @property
    def n_records(self) -> int:
        return len(self.feature_ids)


@dataclass
class PopulationFeatureMatrix:
    """
    Sample-wise feature matrix aggregated across all feature records for a basis.

    Each row is a single usable sample (epoch or stride) and each column is a
    feature type. Structurally padded tensor slots are excluded. `row_class_labels`
    holds the class (faller status) label per row.
    """

    feature_types: List[FeatureType]
    matrix: np.ndarray
    row_class_labels: List[str]

    @property
    def is_empty(self) -> bool:
        return self.matrix.size == 0 or not self.feature_types


@dataclass
class VisualizationDataService:
    """UI-friendly facade over DatabaseManager for visualizations."""

    db_manager: DatabaseManager

    def list_imu_ids(self) -> List[IMUDataIdentifier]:
        return self.db_manager.list_imu_ids()

    def list_feature_ids(self) -> List[FeatureIdentifier]:
        return self.db_manager.list_feature_ids()

    def load_features(self, feature_id: FeatureIdentifier) -> RecordFeatures:
        return self.db_manager.load_features(feature_id)

    def load_imu(self, imu_id: IMUDataIdentifier) -> IMUData:
        return self.db_manager.load_imu(imu_id)

    def load_user_for_imu(self, imu_id: IMUDataIdentifier) -> Optional[UserData]:
        user_id = self.db_manager.get_user_for_imu(imu_id)
        if user_id is None:
            return None
        return self.db_manager.load_user(user_id)

    def load_instrument_spec_for_imu(
        self, imu_id: IMUDataIdentifier
    ) -> Tuple[Optional[InstrumentSpecificationIdentifier], Optional[IMUSpecifications]]:
        spec_id = self.db_manager.get_instrument_spec_for_imu(imu_id)
        if spec_id is None:
            return None, None
        return spec_id, self.db_manager.load_instrument_spec(spec_id)

    def list_feature_types_for_basis(
        self, feature_id: FeatureIdentifier, basis: SampleBasis
    ) -> List[FeatureType]:
        record = self.load_features(feature_id)
        return record.get_features_by_basis(basis).feature_names

    def extract_feature_values(
        self, feature_record: RecordFeatures, basis: SampleBasis, feature_type: FeatureType
    ) -> np.ndarray:
        bout = feature_record.get_features_by_basis(basis)
        if feature_type not in bout.feature_names:
            return np.array([])
        feature_index = bout.feature_names.index(feature_type)
        per_sample = bout.usable_per_sample_matrix()
        if per_sample.size == 0:
            return np.array([])
        column = per_sample[:, feature_index]
        return column[np.isfinite(column)]

    def load_user_for_feature(self, feature_id: FeatureIdentifier) -> Optional[UserData]:
        feature_record = self.load_features(feature_id)
        user_id = feature_record.feature_metadata.user_identifier
        return self.db_manager.load_user(user_id)

    def collect_feature_values_by_class(
        self, basis: SampleBasis, feature_type: FeatureType
    ) -> Dict[str, np.ndarray]:
        grouped: Dict[str, List[float]] = {}
        for feature_id in self.list_feature_ids():
            feature_record = self.load_features(feature_id)
            values = self.extract_feature_values(feature_record, basis, feature_type)
            if values.size == 0:
                continue
            user = self.db_manager.load_user(feature_record.feature_metadata.user_identifier)
            class_label = user.clinical_demographic_data.faller_status.value
            grouped.setdefault(class_label, [])
            grouped[class_label].extend(values.tolist())
        return {
            class_label: np.asarray(samples, dtype=float)
            for class_label, samples in grouped.items()
            if samples
        }

    def collect_population_feature_matrix(
        self,
        basis: SampleBasis,
        feature_types: Optional[List[FeatureType]] = None,
    ) -> PopulationFeatureMatrix:
        """
        Build a sample-wise (rows) by feature-type (columns) matrix across all
        feature records for a basis, aligned to a common feature-type ordering.

        Only *usable* sample rows are included (padding slots in the dense
        ``(bout, feature, sample)`` tensor are excluded).
        """
        reference_types: Optional[List[FeatureType]] = (
            list(feature_types) if feature_types else None
        )
        row_blocks: List[np.ndarray] = []
        labels: List[str] = []

        for feature_id in self.list_feature_ids():
            record = self.load_features(feature_id)
            bout = record.get_features_by_basis(basis)
            record_types = list(bout.feature_names)
            if reference_types is None:
                reference_types = record_types

            aligned, usable = self._align_bout_per_sample(
                bout=bout,
                record_types=record_types,
                reference_types=reference_types,
            )
            if usable.size == 0 or not np.any(usable):
                continue

            user = self.db_manager.load_user(record.feature_metadata.user_identifier)
            label = user.clinical_demographic_data.faller_status.value
            usable_rows = aligned[usable]
            row_blocks.append(usable_rows)
            labels.extend([label] * usable_rows.shape[0])

        if reference_types is None or not row_blocks:
            return PopulationFeatureMatrix(reference_types or [], np.empty((0, 0)), [])

        matrix = np.vstack(row_blocks)
        return PopulationFeatureMatrix(reference_types, matrix, labels)

    def collect_per_record_coverage(self, basis: SampleBasis) -> PerRecordCoverageMatrix:
        """
        Build a record-by-feature matrix of valid-sample fractions.

        Each row summarizes one ``features_*.h5`` file; each column is a
        feature type aligned across the population. Fractions are computed over
        usable sample rows only (tensor padding excluded).
        """
        reference_types: Optional[List[FeatureType]] = None
        row_fractions: List[np.ndarray] = []
        feature_ids: List[str] = []
        participant_ids: List[str] = []
        class_labels: List[str] = []
        usable_counts: List[int] = []

        for feature_id in self.list_feature_ids():
            record = self.load_features(feature_id)
            bout = record.get_features_by_basis(basis)
            record_types = list(bout.feature_names)
            if reference_types is None:
                reference_types = record_types

            user = self.db_manager.load_user(record.feature_metadata.user_identifier)
            participant = record.feature_metadata.user_identifier.value
            class_label = user.clinical_demographic_data.faller_status.value

            aligned, usable = self._align_bout_per_sample(
                bout=bout,
                record_types=record_types,
                reference_types=reference_types or record_types,
            )
            usable_count = int(np.sum(usable)) if usable.size else 0
            usable_counts.append(usable_count)

            if usable_count == 0 or not reference_types:
                fractions = np.full(len(reference_types or record_types), np.nan)
            else:
                populated = aligned[usable]
                fractions = np.array(
                    [
                        float(np.isfinite(populated[:, col]).mean())
                        for col in range(populated.shape[1])
                    ],
                    dtype=float,
                )

            row_fractions.append(fractions)
            feature_ids.append(feature_id.value)
            participant_ids.append(participant)
            class_labels.append(class_label)

        if reference_types is None:
            return PerRecordCoverageMatrix([], np.empty((0, 0)), [], [], [], [])

        matrix = np.vstack(row_fractions) if row_fractions else np.empty((0, len(reference_types)))
        return PerRecordCoverageMatrix(
            feature_types=reference_types,
            matrix=matrix,
            feature_ids=feature_ids,
            participant_ids=participant_ids,
            class_labels=class_labels,
            usable_sample_counts=usable_counts,
        )

    def collect_sample_counts_by_basis_class(
        self,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Per-participant usable sample counts grouped by sampling basis and class.

        Returns ``{basis_value: {class_label: ndarray of per-participant counts}}``.
        A participant's count is the number of samples (across all of their
        feature records) that have at least one non-NaN feature, i.e. samples
        that would survive into a classification run. Participants with a record
        but no usable samples for a basis contribute a count of zero.
        """
        per_basis_counts: Dict[SampleBasis, Dict[str, int]] = {
            SampleBasis.EPOCH: defaultdict(int),
            SampleBasis.STRIDE: defaultdict(int),
        }
        participant_class: Dict[str, str] = {}

        for feature_id in self.list_feature_ids():
            record = self.load_features(feature_id)
            participant = record.feature_metadata.user_identifier.value
            user = self.db_manager.load_user(record.feature_metadata.user_identifier)
            participant_class[participant] = user.clinical_demographic_data.faller_status.value

            for basis in (SampleBasis.EPOCH, SampleBasis.STRIDE):
                bout = record.get_features_by_basis(basis)
                # Ensure the participant is registered even with zero usable samples.
                per_basis_counts[basis].setdefault(participant, 0)
                per_basis_counts[basis][participant] += bout.usable_sample_count

        result: Dict[str, Dict[str, np.ndarray]] = {}
        for basis in (SampleBasis.EPOCH, SampleBasis.STRIDE):
            grouped: Dict[str, List[int]] = defaultdict(list)
            for participant, class_label in participant_class.items():
                grouped[class_label].append(per_basis_counts[basis].get(participant, 0))
            result[basis.value] = {
                class_label: np.asarray(values, dtype=float)
                for class_label, values in grouped.items()
            }
        return result

    def get_sql_index_snapshot(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Return UI-ready metadata index contents from SQLite records + relations.
        """
        repository = self.db_manager.repository

        def _records_for_type(id_type: str) -> List[Dict[str, str]]:
            items: List[Dict[str, str]] = []
            for record_id in repository.list_record_ids(id_type):
                path = repository.get_record_path(id_type, record_id)
                items.append(
                    {
                        "id": record_id,
                        "type": id_type,
                        "path": str(path),
                    }
                )
            return items

        imu_records = _records_for_type("imu_data")
        user_records = _records_for_type("user_data")
        feature_records = _records_for_type("feature")
        spec_records = _records_for_type("instrument_specification")

        relations: List[Dict[str, str]] = []
        for imu in imu_records:
            for target_type, target_id in repository.get_targets(
                source_type="imu_data",
                source_id=imu["id"],
            ):
                relations.append(
                    {
                        "source_type": "imu_data",
                        "source_id": imu["id"],
                        "target_type": target_type,
                        "target_id": target_id,
                    }
                )
        for feature in feature_records:
            for target_type, target_id in repository.get_targets(
                source_type="feature",
                source_id=feature["id"],
            ):
                relations.append(
                    {
                        "source_type": "feature",
                        "source_id": feature["id"],
                        "target_type": target_type,
                        "target_id": target_id,
                    }
                )

        return {
            "imu_records": imu_records,
            "user_records": user_records,
            "feature_records": feature_records,
            "instrument_spec_records": spec_records,
            "relations": relations,
        }

    def cleanup_all_features(self) -> int:
        deleted = self.db_manager.cleanup_all_features(delete_payloads=True)
        return len(deleted)

    def build_feature_level_quality_report(self, basis: SampleBasis):
        from src.data_visualization.suite.services.feature_quality_report import (
            FeatureQualityReportBuilder,
        )

        return FeatureQualityReportBuilder(self).build_feature_level(basis)

    def build_participant_level_quality_report(self, basis: SampleBasis):
        from src.data_visualization.suite.services.feature_quality_report import (
            FeatureQualityReportBuilder,
        )

        return FeatureQualityReportBuilder(self).build_participant_level(basis)

    def rollback_last_feature_generation_run(self) -> int:
        # Imported lazily to avoid introducing orchestrator imports across the suite.
        from src.gait_features.gait_feature_dataset_orchestrator import (
            GaitFeatureDatasetOrchestrator,
        )

        orchestrator = GaitFeatureDatasetOrchestrator(db_manager=self.db_manager)
        deleted_count, _ = orchestrator.rollback_last_run_features()
        return deleted_count

    @staticmethod
    def _align_bout_per_sample(
        bout: BoutFeatures,
        record_types: List[Union[FeatureType, StrideFeatureName]],
        reference_types: List[Union[FeatureType, StrideFeatureName]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = np.asarray(bout.features, dtype=float)
        if features.size == 0:
            width = len(reference_types)
            empty = np.empty((0, width), dtype=float)
            return empty, np.empty(0, dtype=bool)

        per_sample = flatten_bout_features(features)
        aligned = np.full((per_sample.shape[0], len(reference_types)), np.nan)
        index_by_type = {ftype: idx for idx, ftype in enumerate(record_types)}
        for col, ftype in enumerate(reference_types):
            source_index = index_by_type.get(ftype)
            if source_index is not None:
                aligned[:, col] = per_sample[:, source_index]
        return aligned, usable_sample_mask(aligned)
