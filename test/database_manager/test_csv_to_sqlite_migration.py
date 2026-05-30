import csv
import tempfile
import unittest
from pathlib import Path

from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.database_manager.migrations.csv_to_sqlite import migrate_csv_indexes
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class TestCSVToSQLiteMigration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "index.db"
        self.repo = MetadataRepository(SQLiteStore(self.db_path))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_csv(self, path: Path, headers: list[str], rows: list[list[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_migrate_registry_and_mapping_csvs(self):
        imu_registry_dir = self.root / "registries" / "imu_data"
        user_registry_dir = self.root / "registries" / "user_data"
        feature_registry_dir = self.root / "registries" / "feature"
        imu_to_user_dir = self.root / "mappings" / "imu_to_user"
        feature_to_imu_dir = self.root / "mappings" / "feature_to_imu"

        self._write_csv(
            imu_registry_dir / "registry.csv",
            ["data_identifier", "directory"],
            [["imu_1", str(self.root / "imu_data_1")]],
        )
        self._write_csv(
            user_registry_dir / "registry.csv",
            ["data_identifier", "directory"],
            [["user_1", str(self.root / "user_data_1")]],
        )
        self._write_csv(
            feature_registry_dir / "registry.csv",
            ["data_identifier", "directory"],
            [["feature_1", str(self.root / "feature_1")]],
        )
        self._write_csv(
            imu_to_user_dir / "mapping.csv",
            ["source_data_identifier", "target_data_identifier"],
            [["imu_1", "user_1"]],
        )
        self._write_csv(
            feature_to_imu_dir / "mapping.csv",
            ["source_data_identifier", "target_data_identifier"],
            [["feature_1", "imu_1"]],
        )

        migrate_csv_indexes(
            repository=self.repo,
            registry_paths={
                IMUDataIdentifier: imu_registry_dir,
                UserIdentifier: user_registry_dir,
                FeatureIdentifier: feature_registry_dir,
            },
            mapping_paths={
                (IMUDataIdentifier, UserIdentifier): imu_to_user_dir,
                (FeatureIdentifier, IMUDataIdentifier): feature_to_imu_dir,
            },
        )

        self.assertTrue(self.repo.record_exists("imu_data", "imu_1"))
        self.assertTrue(self.repo.record_exists("user_data", "user_1"))
        self.assertTrue(self.repo.record_exists("feature", "feature_1"))

        self.assertTrue(
            self.repo.relation_exists(
                source_type="imu_data",
                source_id="imu_1",
                target_type="user_data",
                target_id="user_1",
                relation_type="imu_to_user",
            )
        )
        self.assertTrue(
            self.repo.relation_exists(
                source_type="feature",
                source_id="feature_1",
                target_type="imu_data",
                target_id="imu_1",
                relation_type="feature_to_imu",
            )
        )


if __name__ == "__main__":
    unittest.main()
