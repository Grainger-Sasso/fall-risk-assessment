import tempfile
import unittest
from pathlib import Path

from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore


class TestSQLiteMetadataRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "index.db"
        self.store = SQLiteStore(self.db_path)
        self.repo = MetadataRepository(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_upsert_and_get_record_path(self):
        record_path = Path(self.temp_dir.name) / "imu_data_123"
        self.repo.upsert_record("imu_data", "imu_1", record_path)

        fetched_path = self.repo.get_record_path("imu_data", "imu_1")
        self.assertEqual(fetched_path, record_path)
        self.assertTrue(self.repo.record_exists("imu_data", "imu_1"))
        self.assertEqual(self.repo.list_record_ids("imu_data"), ["imu_1"])

    def test_add_and_query_relations(self):
        self.repo.add_relation(
            source_type="feature",
            source_id="feature_1",
            target_type="imu_data",
            target_id="imu_1",
            relation_type="feature_to_imu",
        )
        targets = self.repo.get_targets("feature", "feature_1")
        self.assertEqual(targets, [("imu_data", "imu_1")])

        sources = self.repo.get_sources("imu_data", "imu_1")
        self.assertEqual(sources, [("feature", "feature_1")])

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
