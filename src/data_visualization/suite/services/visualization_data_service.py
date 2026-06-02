from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.record_features import RecordFeatures
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


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
        feature_matrix = bout.features[:, feature_index, :]
        flattened = np.asarray(feature_matrix, dtype=float).reshape(-1)
        return flattened[~np.isnan(flattened)]

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
