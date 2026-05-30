import shutil
from pathlib import Path
from typing import Optional, Type, TypeVar

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.record_features import RecordFeatures
from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_errors import (
    DatabaseReadError,
    DatabaseWriteError,
    MetadataIndexError,
)
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.relation_types import FEATURE_TO_IMU, IMU_TO_USER
from src.database_manager.metadata.type_registry import IdentifierTypeRegistry
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.identifier import Identifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier

IdentifierT = TypeVar("IdentifierT", bound=Identifier)


class DatabaseManager:
    def __init__(
        self,
        metadata_repository: MetadataRepository,
        io_router: DomainIORouter,
    ):
        self.repository = metadata_repository
        self.io = io_router

    def save_imu(self, imu_data: IMUData) -> Path:
        source_id = imu_data.metadata.imu_data_identifier
        user_id = imu_data.metadata.user_identifier
        output_path: Optional[Path] = None
        try:
            output_path = self.io.export_imu(imu_data)
            self._upsert_record(source_id, output_path)
            self._add_relation(source_id, user_id, IMU_TO_USER)
            return output_path
        except Exception as exc:
            self._rollback_output(output_path)
            raise DatabaseWriteError(
                f"Failed to save IMU data {source_id.value}: {exc}"
            ) from exc

    def save_features(self, record_features: RecordFeatures) -> Path:
        source_id = record_features.feature_metadata.feature_identifier
        imu_id = record_features.feature_metadata.imu_data_identifier
        output_path: Optional[Path] = None
        try:
            output_path = self.io.export_features(record_features)
            self._upsert_record(source_id, output_path)
            self._add_relation(source_id, imu_id, FEATURE_TO_IMU)
            return output_path
        except Exception as exc:
            self._rollback_output(output_path)
            raise DatabaseWriteError(
                f"Failed to save feature data {source_id.value}: {exc}"
            ) from exc

    def load_imu(self, imu_id: IMUDataIdentifier) -> IMUData:
        try:
            directory = self._get_record_path(imu_id)
            return self.io.import_imu(imu_id, directory)
        except Exception as exc:
            raise DatabaseReadError(
                f"Failed to load IMU data for {imu_id.value}: {exc}"
            ) from exc

    def load_user(self, user_id: UserIdentifier) -> UserData:
        try:
            directory = self._get_record_path(user_id)
            user_data = self.io.import_user(directory)
            if user_data.user_metadata.user_identifier.value != user_id.value:
                raise ValueError(
                    "User payload identifier does not match requested identifier: "
                    f"{user_id.value}"
                )
            return user_data
        except Exception as exc:
            raise DatabaseReadError(
                f"Failed to load user data for {user_id.value}: {exc}"
            ) from exc

    def load_features(self, feature_id: FeatureIdentifier) -> RecordFeatures:
        try:
            directory = self._get_record_path(feature_id)
            return self.io.import_features(feature_id, directory)
        except Exception as exc:
            raise DatabaseReadError(
                f"Failed to load feature data for {feature_id.value}: {exc}"
            ) from exc

    def list_imu_ids(self) -> list[IMUDataIdentifier]:
        return self._list_ids(IMUDataIdentifier)

    def list_feature_ids(self) -> list[FeatureIdentifier]:
        return self._list_ids(FeatureIdentifier)

    def list_user_ids(self) -> list[UserIdentifier]:
        return self._list_ids(UserIdentifier)

    def get_user_for_imu(self, imu_id: IMUDataIdentifier) -> Optional[UserIdentifier]:
        target = self._get_single_target(imu_id, IMU_TO_USER)
        if target is None:
            return None
        if not isinstance(target, UserIdentifier):
            raise MetadataIndexError(
                "Expected user identifier for IMU relation, found "
                f"{type(target).__name__}"
            )
        return target

    def get_imu_for_feature(
        self, feature_id: FeatureIdentifier
    ) -> Optional[IMUDataIdentifier]:
        target = self._get_single_target(feature_id, FEATURE_TO_IMU)
        if target is None:
            return None
        if not isinstance(target, IMUDataIdentifier):
            raise MetadataIndexError(
                "Expected IMU identifier for feature relation, found "
                f"{type(target).__name__}"
            )
        return target

    def get_features_for_imu(self, imu_id: IMUDataIdentifier) -> list[FeatureIdentifier]:
        source_type = IdentifierTypeRegistry.get_type_name(FeatureIdentifier)
        target_type = IdentifierTypeRegistry.get_type_name(IMUDataIdentifier)
        rows = self.repository.get_sources(
            target_type=target_type,
            target_id=imu_id.value,
            relation_type=FEATURE_TO_IMU,
        )
        feature_ids: list[FeatureIdentifier] = []
        for source_type_name, source_id_value in rows:
            if source_type_name != source_type:
                continue
            try:
                identifier = IdentifierTypeRegistry.build_identifier(
                    source_type_name, source_id_value
                )
            except Exception as exc:
                raise MetadataIndexError(
                    "Failed to construct source identifier for feature relation: "
                    f"{source_id_value}"
                ) from exc
            if isinstance(identifier, FeatureIdentifier):
                feature_ids.append(identifier)
        return feature_ids

    def _get_record_path(self, identifier: Identifier) -> Path:
        id_type = IdentifierTypeRegistry.get_type_name(type(identifier))
        try:
            return self.repository.get_record_path(id_type, identifier.value)
        except Exception as exc:
            raise MetadataIndexError(
                f"Failed to resolve record path for {id_type}:{identifier.value}"
            ) from exc

    def _upsert_record(self, identifier: Identifier, path: Path) -> None:
        id_type = IdentifierTypeRegistry.get_type_name(type(identifier))
        try:
            self.repository.upsert_record(id_type, identifier.value, path)
        except Exception as exc:
            raise MetadataIndexError(
                f"Failed to upsert record for {id_type}:{identifier.value}"
            ) from exc

    def _add_relation(
        self, source_id: Identifier, target_id: Identifier, relation_type: str
    ) -> None:
        source_type = IdentifierTypeRegistry.get_type_name(type(source_id))
        target_type = IdentifierTypeRegistry.get_type_name(type(target_id))
        try:
            self.repository.add_relation(
                source_type=source_type,
                source_id=source_id.value,
                target_type=target_type,
                target_id=target_id.value,
                relation_type=relation_type,
            )
        except Exception as exc:
            raise MetadataIndexError(
                "Failed to add relation "
                f"{relation_type}: {source_type}:{source_id.value} -> "
                f"{target_type}:{target_id.value}"
            ) from exc

    def _list_ids(self, id_type: Type[IdentifierT]) -> list[IdentifierT]:
        id_type_name = IdentifierTypeRegistry.get_type_name(id_type)
        try:
            return [
                id_type(value) for value in self.repository.list_record_ids(id_type_name)
            ]
        except Exception as exc:
            raise MetadataIndexError(
                f"Failed to list identifiers for type {id_type_name}"
            ) from exc

    def _get_single_target(
        self, source_id: Identifier, relation_type: str
    ) -> Optional[Identifier]:
        source_type = IdentifierTypeRegistry.get_type_name(type(source_id))
        try:
            targets = self.repository.get_targets(
                source_type=source_type,
                source_id=source_id.value,
                relation_type=relation_type,
            )
        except Exception as exc:
            raise MetadataIndexError(
                f"Failed to fetch relation targets for {source_type}:{source_id.value}"
            ) from exc

        if not targets:
            return None
        target_type_name, target_id_value = targets[0]
        try:
            return IdentifierTypeRegistry.build_identifier(target_type_name, target_id_value)
        except Exception as exc:
            raise MetadataIndexError(
                f"Unknown target identifier type in metadata: {target_type_name}"
            ) from exc

    @staticmethod
    def _rollback_output(output_path: Optional[Path]) -> None:
        if output_path and output_path.exists():
            try:
                shutil.rmtree(output_path)
            except Exception:
                pass
