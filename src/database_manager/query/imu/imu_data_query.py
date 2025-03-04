from typing import Dict, List

from src.data_model.data.imu.imu_data import IMUData
from src.database_manager.query.query_interface import RelatedDataQueryInterface
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class IMUDataQuery(RelatedDataQueryInterface[IMUData, IMUDataIdentifier]):
    """Interface for querying IMU data"""

    def get_imu_data(self, imu_id: IMUDataIdentifier) -> IMUData:
        """Get IMU data by ID

        Args:
            imu_id: IMU data identifier

        Returns:
            IMU data object

        Raises:
            KeyError: If IMU data not found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data(imu_id)

    def get_all_imu_data(self) -> Dict[IMUDataIdentifier, IMUData]:
        """Get all IMU data

        Returns:
            Dictionary mapping IMU IDs to IMU data objects

        Raises:
            FileNotFoundError: If any data file not found
            ImportError: If any data import fails
        """
        return self.get_all_data()

    def get_user_imu_data(self, user_id: UserIdentifier) -> List[IMUData]:
        """Get all IMU data for a user

        Args:
            user_id: User identifier

        Returns:
            List of IMU data objects for the user

        Raises:
            KeyError: If no IMU data found for user
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data_for_target(user_id) 