import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.database_manager.database_manager import DatabaseManager
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier

LAST_RUN_SUMMARY_FILENAME = "feature_generation_last_run.json"


@dataclass
class FeatureGenerationRunSummary:
    status: str = "not_started"
    attempted_imu_ids: List[IMUDataIdentifier] = field(default_factory=list)
    generated_feature_ids: List[FeatureIdentifier] = field(default_factory=list)
    failed_imu_ids: List[IMUDataIdentifier] = field(default_factory=list)
    errors_by_imu_id: Dict[str, str] = field(default_factory=dict)
    stride_failure_imu_ids: List[IMUDataIdentifier] = field(default_factory=list)
    rollback_performed: bool = False
    rolled_back_feature_ids: List[FeatureIdentifier] = field(default_factory=list)
    extraction_backend: str = GaitExtractionBackendId.SKDH.value
    extraction_profile: str = ExtractionProfile.FREE_LIVING.value

    @property
    def succeeded_count(self) -> int:
        return len(self.generated_feature_ids)

    @property
    def failed_count(self) -> int:
        return len(self.failed_imu_ids)

    def to_payload(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempted_imu_ids": [item.value for item in self.attempted_imu_ids],
            "generated_feature_ids": [item.value for item in self.generated_feature_ids],
            "failed_imu_ids": [item.value for item in self.failed_imu_ids],
            "stride_failure_imu_ids": [item.value for item in self.stride_failure_imu_ids],
            "errors_by_imu_id": self.errors_by_imu_id,
            "rollback_performed": self.rollback_performed,
            "rolled_back_feature_ids": [item.value for item in self.rolled_back_feature_ids],
            "extraction_backend": self.extraction_backend,
            "extraction_profile": self.extraction_profile,
        }


class FeatureGenerationRunLog:
    """
    Persists the most recent feature-generation run next to the SQLite index
    so its generated feature records can be rolled back later.
    """

    def __init__(self, db_manager: DatabaseManager, path: Optional[Path] = None):
        self.db_manager = db_manager
        self.path = path or db_manager.sqlite_db_path.parent / LAST_RUN_SUMMARY_FILENAME

    def record(self, summary: FeatureGenerationRunSummary) -> None:
        self._write(summary.to_payload())

    def load(self) -> Optional[Dict[str, object]]:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return None
        return data

    def rollback_last_run(self) -> Tuple[int, List[FeatureIdentifier]]:
        payload = self.load()
        if payload is None:
            return 0, []

        if payload.get("rollback_performed", False):
            return 0, []

        generated_values = payload.get("generated_feature_ids", [])
        generated_ids = [FeatureIdentifier(item) for item in generated_values]
        deleted_feature_ids = self.db_manager.delete_features_by_ids(
            generated_ids,
            delete_payloads=True,
        )
        payload["rollback_performed"] = len(deleted_feature_ids) > 0
        payload["rolled_back_feature_ids"] = [item.value for item in deleted_feature_ids]
        if payload["rollback_performed"] and payload.get("status") == "partial_success":
            payload["status"] = "partial_success_rolled_back"
        self._write(payload)
        return len(deleted_feature_ids), deleted_feature_ids

    def _write(self, payload: Dict[str, object]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
