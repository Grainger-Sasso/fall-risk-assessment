from dataclasses import dataclass
from pathlib import Path

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.exporters.feature.feature_file_exporter import (
    FeatureFileExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import IMUDataImporter
from src.data_io.import_export.importers.data.user.user_data_importer import UserDataImporter
from src.data_io.import_export.importers.features.feature_importer import FeatureImporter
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.record_features import RecordFeatures
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class DomainIORouter:
    """Domain-typed import/export gateway for database manager operations."""

    imu_importer: IMUDataImporter
    user_importer: UserDataImporter
    feature_importer: FeatureImporter
    imu_exporter: IMUDataFileExporter
    feature_exporter: FeatureFileExporter
    imu_output_dir: Path
    feature_output_dir: Path

    def export_imu(self, imu_data: IMUData) -> Path:
        return self.imu_exporter.export_data(self.imu_output_dir, imu_data)

    def export_features(self, record_features: RecordFeatures) -> Path:
        return self.feature_exporter.export_data(self.feature_output_dir, record_features)

    def import_imu(self, identifier: IMUDataIdentifier, directory: Path) -> IMUData:
        data = self.imu_importer.import_data(directory)
        if data.metadata.imu_data_identifier.value != identifier.value:
            raise ValueError(
                "IMU payload identifier does not match requested identifier: "
                f"{identifier.value}"
            )
        return data

    def import_user(self, directory: Path) -> UserData:
        return self.user_importer.import_data(directory)

    def import_features(
        self, identifier: FeatureIdentifier, directory: Path
    ) -> RecordFeatures:
        data = self.feature_importer.import_data(directory)
        if data.feature_metadata.feature_identifier.value != identifier.value:
            raise ValueError(
                "Feature payload identifier does not match requested identifier: "
                f"{identifier.value}"
            )
        return data
