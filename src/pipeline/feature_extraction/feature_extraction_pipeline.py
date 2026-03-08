from pathlib import Path

from src.data_io.import_export.importers.dataset.dataset_importer import DatasetImporter
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.dataset.dataset import Dataset
from src.gait_features.feature_extraction.gait_feature_extractor import (
    GaitFeatureExtractor,
    GaitResults,
)
from src.gait_features.feature_processing.gait_feature_processor import (
    GaitFeatureProcessor,
)
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class FeatureExtractionPipeline:

    def __init__(
        self,
        db_manager: DatabaseManager
    ):
        self.db_manager: DatabaseManager = db_manager
        self.gait_feature_extractor = GaitFeatureExtractor()
        self.gait_feature_processor = GaitFeatureProcessor()
        self.dataset_importer = DatasetImporter()

    def run(self, dataset_path: Path, dataset_name: str):
        dataset: Dataset = self.dataset_importer.import_data(dataset_path, dataset_name)
        for entry in dataset.entries:
            imu_data_id: IMUDataIdentifier = entry.imu_data_id
            user_data_id: UserIdentifier = entry.user_data_id
            imu_data: IMUData = self.db_manager.import_data([imu_data_id])[0]
            user_data: UserData = self.db_manager.import_data([user_data_id])[0]
            gait_res: GaitResults = self.gait_feature_extractor.extract_gait_features(
                imu_data, user_data
            )
            raw_features, agg_features = self.gait_feature_processor.process_features(
                gait_res, imu_data, user_data
            )
            print(f'Exporting gait features for user [{user_data_id.value}]')
            self.db_manager.export_data([raw_features])
            self.db_manager.export_data([agg_features])

