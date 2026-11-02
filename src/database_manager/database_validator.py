from src.database_manager.database_errors import (
    DatabaseReadError,
    DatabaseValidationError,
)
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class DatabaseValidator:
    """Check that indexed records agree with their payloads on disk.

    Each ``validate_*`` method checks every record of its type, then raises a
    single ``DatabaseValidationError`` listing all issues found. IMU payloads are
    read header-only, so no sensor arrays are loaded.
    """

    def validate_imu_data(self, db_manager: DatabaseManager) -> bool:
        issues: list[str] = []
        indexed_user_ids = set(db_manager.list_user_ids())
        checked_users: set[str] = set()

        for imu_id in db_manager.list_imu_ids():
            try:
                metadata = db_manager.load_imu_metadata(imu_id)
            except DatabaseReadError as exc:
                issues.append(f"IMU {imu_id.value}: {exc}")
                continue

            user_id = metadata.user_identifier
            if user_id not in indexed_user_ids:
                issues.append(
                    f"IMU {imu_id.value}: user {user_id.value} from payload is not indexed"
                )
                continue

            linked_user = db_manager.get_user_for_imu(imu_id)
            if linked_user is None or linked_user.value != user_id.value:
                linked = "no user" if linked_user is None else f"user {linked_user.value}"
                issues.append(
                    f"IMU {imu_id.value}: index links {linked}, payload names user {user_id.value}"
                )

            if user_id.value not in checked_users:
                checked_users.add(user_id.value)
                self._user_loads(db_manager, user_id, issues)

        self._raise_if_issues("IMU data", issues)
        return True

    def validate_feature_data(self, db_manager: DatabaseManager) -> bool:
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

            linked_imu = db_manager.get_imu_for_feature(feature_id)
            if linked_imu is None or linked_imu.value != imu_id.value:
                linked = "no IMU" if linked_imu is None else f"IMU {linked_imu.value}"
                issues.append(
                    f"Feature {feature_id.value}: index links {linked}, "
                    f"payload names IMU {imu_id.value}"
                )

            if imu_id.value not in checked_imus:
                checked_imus.add(imu_id.value)
                self._imu_header_loads(db_manager, imu_id, issues)

        self._raise_if_issues("Feature data", issues)
        return True

    def validate_instrument_spec_data(self, db_manager: DatabaseManager) -> bool:
        issues: list[str] = []
        for spec_id in db_manager.list_instrument_spec_ids():
            try:
                db_manager.load_instrument_spec(spec_id)
            except DatabaseReadError as exc:
                issues.append(f"Instrument spec {spec_id.value}: {exc}")

        self._raise_if_issues("Instrument specification", issues)
        return True

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
