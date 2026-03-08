from pathlib import Path
import unittest

from src.database_manager.database_generator import DatabaseGenerator
from src.database_manager.database_validator import DatabaseValidator
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


def get_database_manager():
    """Construct the same database_manager as in test_feature_extraction_pipeline.py."""
    registry_paths = {
        IMUDataIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/imu_data"
        ),
        UserIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/user_data"
        ),
        RawFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/raw_feature"
        ),
        AggregateFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries/agg_feature"
        ),
    }
    mapping_paths = {
        (IMUDataIdentifier, UserIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/mappings/imu_to_user_mapping"
        ),
        (RawFeatureIdentifier, IMUDataIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/mappings/raw_feat_to_imu_mapping"
        ),
        (AggregateFeatureIdentifier, RawFeatureIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/mappings/agg_feat_to_raw_feat_mapping"
        ),
    }
    output_dir_paths = {
        RawFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/raw_feat"
        ),
        AggregateFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/agg_feat"
        ),
    }
    db_generator = DatabaseGenerator()
    return db_generator.generate_database(
        registry_paths, mapping_paths, output_dir_paths, validate=False
    )


class TestDatabaseValidator(unittest.TestCase):
    def setUp(self):
        self.db_manager = get_database_manager()
        self.validator = DatabaseValidator()

    def test_validate_imu_data(self):
        result = self.validator.validate_imu_data(self.db_manager)
        self.assertTrue(result)

    def test_validate_raw_features(self):
        result = self.validator.validate_raw_features(self.db_manager)
        self.assertTrue(result)

    def test_validate_aggregate_features(self):
        result = self.validator.validate_aggregate_features(self.db_manager)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
