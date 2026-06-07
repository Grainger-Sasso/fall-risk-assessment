import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.exporters.feature.feature_file_exporter import (
    FeatureFileExporter,
)
from src.data_io.import_export.exporters.instrument_specifications.instrument_specification_file_exporter import (
    InstrumentSpecificationFileExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import IMUDataImporter
from src.data_io.import_export.importers.data.user.user_data_importer import UserDataImporter
from src.data_io.import_export.importers.features.feature_importer import FeatureImporter
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.gait_features.gait_feature_module import (
    GaitFeatureExtractor,
    RecordFeatureGenerationBuilder,
    StrideFeatureGenerationError,
)
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

    @property
    def succeeded_count(self) -> int:
        return len(self.generated_feature_ids)

    @property
    def failed_count(self) -> int:
        return len(self.failed_imu_ids)


class GaitFeatureDatasetOrchestrator:
    """
    Orchestrates feature extraction for IMU datasets indexed in DatabaseManager.

    Flow per IMU:
      load IMU/user/spec -> SKDH extraction -> RecordFeatures build -> save_features
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        treadmill_profile: bool = False,
        extractor: Optional[GaitFeatureExtractor] = None,
        record_feature_builder: Optional[RecordFeatureGenerationBuilder] = None,
        epoch_window_seconds: float = 8.0,
        epoch_overlap_seconds: float = 2.0,
    ):
        self.db_manager = db_manager
        self.treadmill_profile = treadmill_profile
        self.extractor = extractor or GaitFeatureExtractor(
            treadmill_profile=treadmill_profile
        )
        self.record_feature_builder = record_feature_builder or RecordFeatureGenerationBuilder(
            window_seconds=epoch_window_seconds,
            overlap_seconds=epoch_overlap_seconds,
        )

    def generate_for_all_imu(
        self,
        continue_on_error: bool = False,
        rollback_on_partial_failure: bool = False,
    ) -> FeatureGenerationRunSummary:
        return self.generate_for_imu_ids(
            imu_ids=self.db_manager.list_imu_ids(),
            continue_on_error=continue_on_error,
            rollback_on_partial_failure=rollback_on_partial_failure,
        )

    def generate_for_imu_ids(
        self,
        imu_ids: List[IMUDataIdentifier],
        continue_on_error: bool = False,
        rollback_on_partial_failure: bool = False,
    ) -> FeatureGenerationRunSummary:
        summary = FeatureGenerationRunSummary(
            status="running",
            attempted_imu_ids=list(imu_ids),
        )
        total = len(imu_ids)
        for index, imu_id in enumerate(imu_ids, start=1):
            print(
                f"[{index}/{total}] Processing IMU record '{imu_id.value}'..."
            )
            try:
                feature_id = self._generate_for_single_imu(imu_id)
                summary.generated_feature_ids.append(feature_id)
                print(
                    f"[{index}/{total}] Completed IMU '{imu_id.value}' -> "
                    f"feature '{feature_id.value}'."
                )
            except StrideFeatureGenerationError as exc:
                diagnosis = getattr(exc, "diagnosis", {})
                print(
                    f"[{index}/{total}] [STRIDE-FAILURE] IMU '{imu_id.value}': "
                    f"stage={diagnosis.get('stage', 'unknown')} :: {exc}"
                )
                print(
                    f"    missing_stride_keys="
                    f"{len(diagnosis.get('missing_stride_keys', []))}/"
                    f"{diagnosis.get('expected_stride_keys', 0)}, "
                    f"bouts_without_events={diagnosis.get('bouts_without_events', [])}, "
                    f"tensor_all_nan={diagnosis.get('tensor_all_nan')}"
                )
                summary.failed_imu_ids.append(imu_id)
                summary.stride_failure_imu_ids.append(imu_id)
                summary.errors_by_imu_id[imu_id.value] = str(exc)
                if not continue_on_error:
                    summary.status = "failed"
                    self._persist_last_run_summary(summary)
                    raise RuntimeError(
                        f"Feature generation failed for IMU '{imu_id.value}': {exc}"
                    ) from exc
                continue
            except Exception as exc:
                print(
                    f"[{index}/{total}] Failed IMU '{imu_id.value}': {exc}"
                )
                summary.failed_imu_ids.append(imu_id)
                summary.errors_by_imu_id[imu_id.value] = str(exc)
                if not continue_on_error:
                    summary.status = "failed"
                    self._persist_last_run_summary(summary)
                    raise RuntimeError(
                        f"Feature generation failed for IMU '{imu_id.value}': {exc}"
                    ) from exc
        summary.status = self._resolve_status(summary)

        if summary.status == "partial_success" and rollback_on_partial_failure:
            deleted_feature_ids = self.db_manager.delete_features_by_ids(
                summary.generated_feature_ids,
                delete_payloads=True,
            )
            summary.rollback_performed = len(deleted_feature_ids) > 0
            summary.rolled_back_feature_ids = deleted_feature_ids
            print(
                "[ROLLBACK] Partial success detected. Removed "
                f"{len(deleted_feature_ids)} newly generated feature records."
            )
            if summary.rollback_performed:
                summary.status = "partial_success_rolled_back"

        self._persist_last_run_summary(summary)
        return summary

    def rollback_last_run_features(self) -> Tuple[int, List[FeatureIdentifier]]:
        payload = self._load_last_run_summary()
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
        self._write_last_run_summary(payload)
        return len(deleted_feature_ids), deleted_feature_ids

    def _resolve_status(self, summary: FeatureGenerationRunSummary) -> str:
        if summary.failed_count == 0:
            return "success"
        if summary.succeeded_count > 0:
            return "partial_success"
        return "failed"

    def _get_last_run_summary_path(self) -> Path:
        sqlite_path = self.db_manager.repository.store.db_path
        return sqlite_path.parent / LAST_RUN_SUMMARY_FILENAME

    def _persist_last_run_summary(self, summary: FeatureGenerationRunSummary) -> None:
        payload = {
            "status": summary.status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempted_imu_ids": [item.value for item in summary.attempted_imu_ids],
            "generated_feature_ids": [item.value for item in summary.generated_feature_ids],
            "failed_imu_ids": [item.value for item in summary.failed_imu_ids],
            "stride_failure_imu_ids": [
                item.value for item in summary.stride_failure_imu_ids
            ],
            "errors_by_imu_id": summary.errors_by_imu_id,
            "rollback_performed": summary.rollback_performed,
            "rolled_back_feature_ids": [
                item.value for item in summary.rolled_back_feature_ids
            ],
        }
        self._write_last_run_summary(payload)

    def _load_last_run_summary(self) -> Optional[Dict[str, object]]:
        run_path = self._get_last_run_summary_path()
        if not run_path.exists():
            return None
        with run_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return None
        return data

    def _write_last_run_summary(self, payload: Dict[str, object]) -> None:
        run_path = self._get_last_run_summary_path()
        with run_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _generate_for_single_imu(self, imu_id: IMUDataIdentifier) -> FeatureIdentifier:
        imu_data = self.db_manager.load_imu(imu_id)
        user_id = self.db_manager.get_user_for_imu(imu_id)
        if user_id is None:
            raise ValueError(f"No user mapping found for IMU '{imu_id.value}'.")
        user_data = self.db_manager.load_user(user_id)

        spec = None
        spec_id = self.db_manager.get_instrument_spec_for_imu(imu_id)
        if spec_id is not None:
            spec = self.db_manager.load_instrument_spec(spec_id)

        gait_results = self.extractor.extract_gait_features(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=spec,
        )
        record_features = self.record_feature_builder.build(
            gait_results=gait_results,
            imu_data=imu_data,
            user_data=user_data,
            treadmill_profile=self.treadmill_profile,
        )
        self.db_manager.save_features(record_features)
        return record_features.feature_metadata.feature_identifier


def build_database_manager_from_sqlite(sqlite_db_path: Path) -> DatabaseManager:
    """
    Build a DatabaseManager against an existing SQLite index for gait generation jobs.
    """
    sqlite_db_path = sqlite_db_path.resolve()
    scratch_dir = sqlite_db_path.parent
    repository = MetadataRepository(SQLiteStore(sqlite_db_path))
    io_router = DomainIORouter(
        imu_importer=IMUDataImporter(),
        user_importer=UserDataImporter(),
        feature_importer=FeatureImporter(),
        instrument_spec_importer=InstrumentSpecificationImporter(),
        imu_exporter=IMUDataFileExporter(),
        feature_exporter=FeatureFileExporter(),
        instrument_spec_exporter=InstrumentSpecificationFileExporter(),
        imu_output_dir=scratch_dir,
        feature_output_dir=scratch_dir,
        instrument_spec_output_dir=scratch_dir,
    )
    return DatabaseManager(metadata_repository=repository, io_router=io_router)


def _parse_imu_ids(raw_value: str) -> List[IMUDataIdentifier]:
    entries = [item.strip() for item in raw_value.split(",") if item.strip()]
    return [IMUDataIdentifier(value) for value in entries]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gait RecordFeatures from IMU/User data indexed in SQLite."
    )
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument(
        "--imu-ids",
        type=str,
        default="",
        help="Optional comma-separated IMU IDs to process. Empty means all indexed IMUs.",
    )
    parser.add_argument(
        "--treadmill-profile",
        action="store_true",
        help="Use treadmill SKDH profile for extraction.",
    )
    parser.add_argument(
        "--epoch-window-seconds",
        type=float,
        default=8.0,
        help="Epoch sliding-window length in seconds (must be between 5 and 10).",
    )
    parser.add_argument(
        "--epoch-overlap-seconds",
        type=float,
        default=2.0,
        help="Epoch sliding-window overlap in seconds (> 0 and < window length).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining IMUs when a record fails.",
    )
    parser.add_argument(
        "--rollback-on-partial-failure",
        action="store_true",
        help=(
            "If some IMUs succeed and others fail, delete newly generated feature "
            "records so the DB returns to its pre-run feature state."
        ),
    )
    parser.add_argument(
        "--rollback-last-run",
        action="store_true",
        help="Rollback feature records generated by the most recent orchestrator run.",
    )
    parser.add_argument(
        "--cleanup-features",
        action="store_true",
        help=(
            "Delete all feature records and feature relations while keeping IMU, user, "
            "and instrument-spec records."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_manager = build_database_manager_from_sqlite(args.sqlite_db_path)
    orchestrator = GaitFeatureDatasetOrchestrator(
        db_manager=db_manager,
        treadmill_profile=args.treadmill_profile,
        epoch_window_seconds=args.epoch_window_seconds,
        epoch_overlap_seconds=args.epoch_overlap_seconds,
    )

    if args.cleanup_features:
        deleted = db_manager.cleanup_all_features(delete_payloads=True)
        print(f"Cleanup complete. Removed {len(deleted)} feature record(s).")
        return

    if args.rollback_last_run:
        deleted_count, deleted_ids = orchestrator.rollback_last_run_features()
        if deleted_count == 0:
            print("No rollback actions were performed for the last run.")
        else:
            print(f"Rollback complete. Removed {deleted_count} feature record(s):")
            for feature_id in deleted_ids:
                print(f"- {feature_id.value}")
        return

    if args.imu_ids.strip():
        imu_ids = _parse_imu_ids(args.imu_ids)
        summary = orchestrator.generate_for_imu_ids(
            imu_ids=imu_ids,
            continue_on_error=args.continue_on_error,
            rollback_on_partial_failure=args.rollback_on_partial_failure,
        )
    else:
        summary = orchestrator.generate_for_all_imu(
            continue_on_error=args.continue_on_error,
            rollback_on_partial_failure=args.rollback_on_partial_failure,
        )

    print(f"Run status: {summary.status}")
    print(f"Generated features: {len(summary.generated_feature_ids)}")
    if summary.generated_feature_ids:
        print("Generated feature IDs:")
        for feature_id in summary.generated_feature_ids:
            print(f"- {feature_id.value}")
    print(f"Failed IMUs: {len(summary.failed_imu_ids)}")
    if summary.failed_imu_ids:
        print("Failures:")
        for imu_id in summary.failed_imu_ids:
            print(f"- {imu_id.value}: {summary.errors_by_imu_id.get(imu_id.value, '')}")
    if summary.stride_failure_imu_ids:
        print(
            f"Stride-feature failures (all-NaN/no strides detected): "
            f"{len(summary.stride_failure_imu_ids)}"
        )
        for imu_id in summary.stride_failure_imu_ids:
            print(f"- {imu_id.value}")
    if summary.rollback_performed:
        print(
            "Rollback complete for this run. Removed "
            f"{len(summary.rolled_back_feature_ids)} feature record(s)."
        )


if __name__ == "__main__":
    main()
