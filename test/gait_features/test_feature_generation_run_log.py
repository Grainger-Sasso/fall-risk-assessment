import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.gait_features.orchestration.feature_generation_run_log import (
    LAST_RUN_SUMMARY_FILENAME,
    FeatureGenerationRunLog,
    FeatureGenerationRunSummary,
)
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class TestFeatureGenerationRunLog(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_manager = MagicMock()
        self.db_manager.sqlite_db_path = Path(self._tmp.name) / "index.db"
        self.run_log = FeatureGenerationRunLog(self.db_manager)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_defaults_to_file_next_to_sqlite_index(self):
        self.assertEqual(self.run_log.path, Path(self._tmp.name) / LAST_RUN_SUMMARY_FILENAME)

    def test_rollback_without_recorded_run_is_noop(self):
        self.assertEqual(self.run_log.rollback_last_run(), (0, []))
        self.db_manager.delete_features_by_ids.assert_not_called()

    def test_rollback_deletes_recorded_features_once(self):
        self.run_log.record(
            FeatureGenerationRunSummary(
                status="partial_success",
                attempted_imu_ids=[IMUDataIdentifier("imu_1"), IMUDataIdentifier("imu_2")],
                generated_feature_ids=[FeatureIdentifier("feature_1")],
                failed_imu_ids=[IMUDataIdentifier("imu_2")],
            )
        )
        self.db_manager.delete_features_by_ids.return_value = [FeatureIdentifier("feature_1")]

        deleted_count, deleted_ids = self.run_log.rollback_last_run()

        self.assertEqual(deleted_count, 1)
        self.assertEqual([item.value for item in deleted_ids], ["feature_1"])
        payload = json.loads(self.run_log.path.read_text(encoding="utf-8"))
        self.assertTrue(payload["rollback_performed"])
        self.assertEqual(payload["status"], "partial_success_rolled_back")
        self.assertEqual(self.run_log.rollback_last_run(), (0, []))
        self.db_manager.delete_features_by_ids.assert_called_once()


if __name__ == "__main__":
    unittest.main()
