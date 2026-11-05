from typing import List, Optional

from src.database_manager.database_manager import DatabaseManager
from src.gait_features.diagnosis.stride_failure import StrideFeatureGenerationError
from src.gait_features.orchestration.feature_generation_run_log import (
    FeatureGenerationRunLog,
    FeatureGenerationRunSummary,
)
from src.gait_features.pipeline.gait_extraction_pipeline import GaitExtractionPipeline
from src.gait_features.processing.backend_requirements import BackendRequirementsError
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class GaitFeatureDatasetOrchestrator:
    """
    Runs a GaitExtractionPipeline over IMU records indexed in DatabaseManager.

    Flow per IMU:
      load IMU/user/spec -> pipeline.run -> save_features
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        pipeline: GaitExtractionPipeline,
        run_log: Optional[FeatureGenerationRunLog] = None,
    ):
        self.db_manager = db_manager
        self.pipeline = pipeline
        self.run_log = run_log or FeatureGenerationRunLog(db_manager)

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
            extraction_backend=self.pipeline.backend.value,
            extraction_profile=self.pipeline.profile.value,
        )
        total = len(imu_ids)
        for index, imu_id in enumerate(imu_ids, start=1):
            progress = f"[{index}/{total}]"
            print(f"{progress} Processing IMU record '{imu_id.value}'...")
            try:
                feature_id = self._generate_for_single_imu(imu_id)
            except Exception as exc:
                self._record_failure(summary, imu_id, exc, progress)
                if not continue_on_error:
                    summary.status = "failed"
                    self.run_log.record(summary)
                    raise RuntimeError(
                        f"Feature generation failed for IMU '{imu_id.value}': {exc}"
                    ) from exc
                continue
            summary.generated_feature_ids.append(feature_id)
            print(
                f"{progress} Completed IMU '{imu_id.value}' -> "
                f"feature '{feature_id.value}'."
            )
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

        self.run_log.record(summary)
        return summary

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

        record_features = self.pipeline.run(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=spec,
        )
        self.db_manager.save_features(record_features)
        return record_features.feature_metadata.feature_identifier

    @staticmethod
    def _record_failure(
        summary: FeatureGenerationRunSummary,
        imu_id: IMUDataIdentifier,
        exc: Exception,
        progress: str,
    ) -> None:
        if isinstance(exc, BackendRequirementsError):
            print(f"{progress} [BACKEND-REQUIREMENTS] IMU '{imu_id.value}': {exc}")
        elif isinstance(exc, StrideFeatureGenerationError):
            diagnosis = getattr(exc, "diagnosis", {})
            print(
                f"{progress} [STRIDE-FAILURE] IMU '{imu_id.value}': "
                f"stage={diagnosis.get('stage', 'unknown')} :: {exc}"
            )
            print(
                f"    missing_stride_keys="
                f"{len(diagnosis.get('missing_stride_keys', []))}/"
                f"{diagnosis.get('expected_stride_keys', 0)}, "
                f"bouts_without_events={diagnosis.get('bouts_without_events', [])}, "
                f"tensor_all_nan={diagnosis.get('tensor_all_nan')}"
            )
            summary.stride_failure_imu_ids.append(imu_id)
        else:
            print(f"{progress} Failed IMU '{imu_id.value}': {exc}")
        summary.failed_imu_ids.append(imu_id)
        summary.errors_by_imu_id[imu_id.value] = str(exc)

    @staticmethod
    def _resolve_status(summary: FeatureGenerationRunSummary) -> str:
        if summary.failed_count == 0:
            return "success"
        if summary.succeeded_count > 0:
            return "partial_success"
        return "failed"
