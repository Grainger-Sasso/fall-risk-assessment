from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.record_features import RecordFeatures
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.identifier import Identifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class DatabaseValidator:
    def __init__(self):
        pass

    def validate_imu_data(self, db_manager: DatabaseManager):
        imu_ids = db_manager.list_imu_ids()
        for imu_identifier in imu_ids:
            try:
                imu_data: IMUData = db_manager.load_imu(imu_identifier)
                imu_id_from_data: Identifier = imu_data.get_data_id()
                user_id: Identifier = imu_data.get_associated_data_id()
                user_id_from_mapping = db_manager.get_user_for_imu(imu_identifier)
                print(f"Validating data for user: {user_id.value}")
                if imu_identifier.value != imu_id_from_data.value:
                    raise ValueError(
                        f"IMU data ID in registry -{imu_identifier.value}- does not match ID in file -{imu_id_from_data.value}-"
                    )
                if user_id not in db_manager.list_user_ids():
                    raise ValueError(
                        f"For IMU ID -{imu_identifier.value}-: User ID not present in registry -{user_id.value}-"
                    )
                if user_id_from_mapping is None or user_id.value != user_id_from_mapping.value:
                    raise ValueError(
                        f"For IMU ID -{imu_identifier.value}-: User ID in mapping does not match ID in file."
                    )
                user_data: UserData = db_manager.load_user(user_id)
                user_id_from_data: Identifier = user_data.get_data_id()
                if user_id_from_data.value != user_id.value:
                    raise ValueError(
                        f"User data identifier from imu data -{user_id.value}- does not match ID in file -{user_id_from_data.value}-"
                    )
            except Exception as e:
                raise Exception(e)
        return True

    def validate_feature_data(self, db_manager: DatabaseManager):
        feature_ids = db_manager.list_feature_ids()
        for feature_identifier in feature_ids:
            try:
                feature_data: RecordFeatures = db_manager.load_features(feature_identifier)
                feature_id_from_data: Identifier = feature_data.get_data_id()
                imu_id: Identifier = feature_data.get_associated_data_id()
                imu_id_from_mapping = db_manager.get_imu_for_feature(feature_identifier)
                print(f"Validating features for IMU data: {imu_id.value}")
                if feature_identifier.value != feature_id_from_data.value:
                    raise ValueError(
                        f"Feature ID in registry -{feature_identifier.value}- does not match ID in file -{feature_id_from_data.value}-"
                    )
                if imu_id not in db_manager.list_imu_ids():
                    raise ValueError(
                        f"For feature ID -{feature_identifier.value}-: IMU ID not present in registry -{imu_id.value}-"
                    )
                if imu_id_from_mapping is None or imu_id.value != imu_id_from_mapping.value:
                    raise ValueError(
                        f"For feature ID -{feature_identifier.value}-: IMU ID in mapping does not match ID in file."
                    )
                imu_data: IMUData = db_manager.load_imu(imu_id)
                imu_id_from_data: Identifier = imu_data.get_data_id()
                if imu_id_from_data.value != imu_id.value:
                    raise ValueError(
                        f"IMU data identifier from feature -{imu_id.value}- does not match ID in file -{imu_id_from_data.value}-"
                    )
            except Exception as e:
                raise Exception(e)
        return True

    # Backward compatibility wrappers for legacy call sites.
    def validate_raw_features(self, db_manager: DatabaseManager):
        return self.validate_feature_data(db_manager)

    def validate_aggregate_features(self, db_manager: DatabaseManager):
        return True