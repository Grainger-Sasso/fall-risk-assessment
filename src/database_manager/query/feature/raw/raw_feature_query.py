from typing import Dict, List

from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.database_manager.query.query_interface import RelatedDataQueryInterface
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class RawFeatureQuery(RelatedDataQueryInterface[RawFeatureSetEntry, RawFeatureIdentifier]):
    """Interface for querying raw feature data"""

    def get_raw_feature(self, feature_id: RawFeatureIdentifier) -> RawFeatureSetEntry:
        """Get raw feature by ID

        Args:
            feature_id: Raw feature identifier

        Returns:
            Raw feature set entry

        Raises:
            KeyError: If feature not found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data(feature_id)

    def get_all_raw_features(self) -> Dict[RawFeatureIdentifier, RawFeatureSetEntry]:
        """Get all raw features

        Returns:
            Dictionary mapping feature IDs to feature objects

        Raises:
            FileNotFoundError: If any data file not found
            ImportError: If any data import fails
        """
        return self.get_all_data()

    def get_imu_features(self, imu_id: IMUDataIdentifier) -> List[RawFeatureSetEntry]:
        """Get all raw features derived from an IMU dataset

        Args:
            imu_id: IMU data identifier

        Returns:
            List of raw feature sets derived from the IMU data

        Raises:
            KeyError: If no features found for IMU data
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data_for_target(imu_id) 