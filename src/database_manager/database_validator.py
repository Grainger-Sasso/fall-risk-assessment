from typing import Optional

from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.database_manager.database_errors import (
    DatabaseReadError,
    DatabaseValidationError,
    MetadataIndexError,
)
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.type_registry import IdentifierTypeRegistry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier


class DatabaseValidator:
    """Check that indexed records agree with their payloads on disk.

    Each ``validate_*`` method checks every record of its scope, then raises a
    single ``DatabaseValidationError`` listing all issues found. IMU payloads are
    read header-only, so no sensor arrays are loaded.
    """

    def validate_database(self, db_manager: DatabaseManager) -> bool:
        """Run every check and report all issues together."""
        issues = (
            self._index_structure_issues(db_manager)
            + self._imu_issues(db_manager)
            + self._feature_issues(db_manager)
            + self._instrument_spec_issues(db_manager)
        )
        self._raise_if_issues("Database", issues)
        return True

    def validate_index_structure(self, db_manager: DatabaseManager) -> bool:
        self._raise_if_issues("Index structure", self._index_structure_issues(db_manager))
        return True

    def validate_imu_data(self, db_manager: DatabaseManager) -> bool:
        self._raise_if_issues("IMU data", self._imu_issues(db_manager))
        return True

    def validate_feature_data(self, db_manager: DatabaseManager) -> bool:
        self._raise_if_issues("Feature data", self._feature_issues(db_manager))
        return True

    def validate_instrument_spec_data(self, db_manager: DatabaseManager) -> bool:
        self._raise_if_issues(
            "Instrument specification", self._instrument_spec_issues(db_manager)
        )
        return True

    @staticmethod
    def _index_structure_issues(db_manager: DatabaseManager) -> list[str]:
        repository = db_manager.repository
        issues: list[str] = []
        for id_type, count in repository.count_records_by_type().items():
            if id_type not in IdentifierTypeRegistry.NAME_TO_TYPE:
                issues.append(f"{count} record(s) have unknown type '{id_type}'")

        for source_type, target_type, relation_type, count in (
            repository.count_relations_by_kind()
        ):
            expected = IdentifierTypeRegistry.DEFAULT_RELATIONS.get((source_type, target_type))
            if relation_type != expected:
                issues.append(
                    f"{count} '{relation_type}' relation(s) from {source_type} to "
                    f"{target_type} are not an allowed relation"
                )

        for source_type, source_id, relation_type, count in (
            repository.find_sources_with_multiple_targets()
        ):
            issues.append(
                f"{source_type} {source_id} has {count} '{relation_type}' relations; "
                "expected at most one"
            )
        return issues

    def _imu_issues(self, db_manager: DatabaseManager) -> list[str]:
        issues: list[str] = []
        indexed_user_ids = set(db_manager.list_user_ids())
        spec_ids = db_manager.list_instrument_spec_ids()
        spec_names: Optional[dict[str, str]] = None
        checked_users: set[str] = set()

        for imu_id in db_manager.list_imu_ids():
            try:
                metadata = db_manager.load_imu_metadata(imu_id)
            except DatabaseReadError as exc:
                issues.append(f"IMU {imu_id.value}: {exc}")
                continue

            if len(spec_ids) > 1 and spec_names is None:
                spec_names = self._load_spec_names(db_manager, spec_ids)
            spec_issue = self._spec_link_issue(
                db_manager, imu_id, metadata, spec_ids, spec_names or {}
            )
            if spec_issue is not None:
                issues.append(spec_issue)

            user_id = metadata.user_identifier
            if user_id not in indexed_user_ids:
                issues.append(
                    f"IMU {imu_id.value}: user {user_id.value} from payload is not indexed"
                )
                continue

            try:
                linked_user = db_manager.get_user_for_imu(imu_id)
            except MetadataIndexError as exc:
                issues.append(f"IMU {imu_id.value}: {exc}")
                continue
            if linked_user is None or linked_user.value != user_id.value:
                linked = "no user" if linked_user is None else f"user {linked_user.value}"
                issues.append(
                    f"IMU {imu_id.value}: index links {linked}, payload names user {user_id.value}"
                )

            if user_id.value not in checked_users:
                checked_users.add(user_id.value)
                self._user_loads(db_manager, user_id, issues)
        return issues

    def _feature_issues(self, db_manager: DatabaseManager) -> list[str]:
        issues: list[str] = []
        indexed_imu_ids = set(db_manager.list_imu_ids())
        checked_imus: set[str] = set()

        for feature_id in db_manager.list_feature_ids():
            try:
                feature_data = db_manager.load_features(feature_id)
            except DatabaseReadError as exc:
                issues.append(f"Feature {feature_id.value}: {exc}")
                continue

            imu_id = feature_data.get_associated_data_id()
            if imu_id not in indexed_imu_ids:
                issues.append(
                    f"Feature {feature_id.value}: IMU {imu_id.value} from payload is not indexed"
                )
                continue

            try:
                linked_imu = db_manager.get_imu_for_feature(feature_id)
            except MetadataIndexError as exc:
                issues.append(f"Feature {feature_id.value}: {exc}")
                continue
            if linked_imu is None or linked_imu.value != imu_id.value:
                linked = "no IMU" if linked_imu is None else f"IMU {linked_imu.value}"
                issues.append(
                    f"Feature {feature_id.value}: index links {linked}, "
                    f"payload names IMU {imu_id.value}"
                )

            if imu_id.value not in checked_imus:
                checked_imus.add(imu_id.value)
                self._imu_header_loads(db_manager, imu_id, issues)
        return issues

    @staticmethod
    def _instrument_spec_issues(db_manager: DatabaseManager) -> list[str]:
        issues: list[str] = []
        for spec_id in db_manager.list_instrument_spec_ids():
            try:
                db_manager.load_instrument_spec(spec_id)
            except DatabaseReadError as exc:
                issues.append(f"Instrument spec {spec_id.value}: {exc}")
        return issues

    @staticmethod
    def _spec_link_issue(
        db_manager: DatabaseManager,
        imu_id: IMUDataIdentifier,
        metadata: IMUMetadata,
        spec_ids: list[InstrumentSpecificationIdentifier],
        spec_names: dict[str, str],
    ) -> Optional[str]:
        """Apply the build rule: one spec links every IMU; otherwise match by instrument name.

        ``spec_names`` maps loadable spec IDs to normalized names; unloadable specs
        are reported by the instrument spec check, so they are skipped here.
        """
        try:
            linked_spec = db_manager.get_instrument_spec_for_imu(imu_id)
        except MetadataIndexError as exc:
            return f"IMU {imu_id.value}: {exc}"

        if len(spec_ids) == 1:
            if linked_spec is None:
                return (
                    f"IMU {imu_id.value}: not linked to the only indexed instrument spec "
                    f"{spec_ids[0].value}"
                )
            return None

        instrument_name = _normalize_name(metadata.instrument_identifier.name)
        expected = [spec for spec, name in spec_names.items() if name == instrument_name]
        if linked_spec is None:
            if expected:
                return (
                    f"IMU {imu_id.value}: not linked to instrument spec {expected[0]}, "
                    f"which matches instrument '{metadata.instrument_identifier.name}'"
                )
            return None
        if linked_spec.value in spec_names and linked_spec.value not in expected:
            return (
                f"IMU {imu_id.value}: linked to instrument spec {linked_spec.value}, "
                f"which does not match instrument '{metadata.instrument_identifier.name}'"
            )
        return None

    @staticmethod
    def _load_spec_names(
        db_manager: DatabaseManager, spec_ids: list[InstrumentSpecificationIdentifier]
    ) -> dict[str, str]:
        names: dict[str, str] = {}
        for spec_id in spec_ids:
            try:
                names[spec_id.value] = _normalize_name(
                    db_manager.load_instrument_spec(spec_id).imu_name
                )
            except DatabaseReadError:
                continue
        return names

    @staticmethod
    def _user_loads(
        db_manager: DatabaseManager, user_id: UserIdentifier, issues: list[str]
    ) -> None:
        try:
            db_manager.load_user(user_id)
        except DatabaseReadError as exc:
            issues.append(f"User {user_id.value}: {exc}")

    @staticmethod
    def _imu_header_loads(
        db_manager: DatabaseManager, imu_id: IMUDataIdentifier, issues: list[str]
    ) -> None:
        try:
            db_manager.load_imu_metadata(imu_id)
        except DatabaseReadError as exc:
            issues.append(f"IMU {imu_id.value}: {exc}")

    @staticmethod
    def _raise_if_issues(scope: str, issues: list[str]) -> None:
        if issues:
            raise DatabaseValidationError(scope, issues)


def _normalize_name(name: str) -> str:
    return name.strip().lower()
