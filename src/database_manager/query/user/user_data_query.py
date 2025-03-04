from src.data_model.data.user.user_data import UserData
from src.database_manager.query.query_interface import QueryInterface
from src.identifiers.user.user_identifier import UserIdentifier
from typing import Dict


class UserDataQuery(QueryInterface[UserData, UserIdentifier]):
    """Interface for querying user data"""

    def get_user(self, user_id: UserIdentifier) -> UserData:
        """Get user data by ID

        Args:
            user_id: User identifier

        Returns:
            User data object

        Raises:
            KeyError: If user not found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data(user_id)

    def get_all_users(self) -> Dict[UserIdentifier, UserData]:
        """Get all user data

        Returns:
            Dictionary mapping user IDs to user data objects

        Raises:
            FileNotFoundError: If any data file not found
            ImportError: If any data import fails
        """
        return self.get_all_data() 