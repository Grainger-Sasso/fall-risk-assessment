import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.gait_features.gait_feature_dataset_orchestrator import (
    GaitFeatureDatasetOrchestrator,
)
from src.gait_features.gait_feature_module import StrideFeatureGenerationError
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class TestGaitFeatureDatasetOrchestrator(unittest.TestCase):
    def test_generate_for_all_imu_happy_path(self):
        db_manager = MagicMock()
        imu_ids = [IMUDataIdentifier("imu_1"), IMUDataIdentifier("imu_2")]
        db_manager.list_imu_ids.return_value = imu_ids
        db_manager.get_user_for_imu.side_effect = [
            UserIdentifier("user_1"),
            UserIdentifier("user_2"),
        ]
        db_manager.load_imu.side_effect = ["imu_data_1", "imu_data_2"]
        db_manager.load_user.side_effect = ["user_data_1", "user_data_2"]
        db_manager.get_instrument_spec_for_imu.return_value = None

        extractor = MagicMock()
        extractor.extract_gait_features.side_effect = ["gait_results_1", "gait_results_2"]
        builder = MagicMock()
        builder.build.side_effect = [
            SimpleNamespace(
                feature_metadata=SimpleNamespace(feature_identifier=SimpleNamespace(value="feature_1"))
            ),
            SimpleNamespace(
                feature_metadata=SimpleNamespace(feature_identifier=SimpleNamespace(value="feature_2"))
            ),
        ]

        orchestrator = GaitFeatureDatasetOrchestrator(
            db_manager=db_manager,
            treadmill_profile=True,
            extractor=extractor,
            record_feature_builder=builder,
        )
        orchestrator._write_last_run_summary = MagicMock()
        summary = orchestrator.generate_for_all_imu()

        self.assertEqual([item.value for item in summary.generated_feature_ids], ["feature_1", "feature_2"])
        self.assertEqual(summary.failed_imu_ids, [])
        self.assertEqual(summary.errors_by_imu_id, {})
        self.assertEqual(summary.status, "success")
        self.assertEqual(db_manager.save_features.call_count, 2)

    def test_continue_on_error_records_failed_ids(self):
        db_manager = MagicMock()
        imu_ids = [IMUDataIdentifier("imu_1"), IMUDataIdentifier("imu_2")]
        db_manager.list_imu_ids.return_value = imu_ids
        db_manager.load_imu.side_effect = ["imu_data_1", RuntimeError("cannot load")]
        db_manager.get_user_for_imu.return_value = UserIdentifier("user_1")
        db_manager.load_user.return_value = "user_data_1"
        db_manager.get_instrument_spec_for_imu.return_value = None

        extractor = MagicMock()
        extractor.extract_gait_features.return_value = "gait_results_1"
        builder = MagicMock()
        builder.build.return_value = SimpleNamespace(
            feature_metadata=SimpleNamespace(feature_identifier=SimpleNamespace(value="feature_1"))
        )

        orchestrator = GaitFeatureDatasetOrchestrator(
            db_manager=db_manager,
            extractor=extractor,
            record_feature_builder=builder,
        )
        orchestrator._write_last_run_summary = MagicMock()
        summary = orchestrator.generate_for_all_imu(continue_on_error=True)

        self.assertEqual([item.value for item in summary.generated_feature_ids], ["feature_1"])
        self.assertEqual([item.value for item in summary.failed_imu_ids], ["imu_2"])
        self.assertIn("imu_2", summary.errors_by_imu_id)
        self.assertEqual(summary.status, "partial_success")

    def test_partial_success_with_rollback_removes_generated_features(self):
        db_manager = MagicMock()
        imu_ids = [IMUDataIdentifier("imu_1"), IMUDataIdentifier("imu_2")]
        db_manager.list_imu_ids.return_value = imu_ids
        db_manager.load_imu.side_effect = ["imu_data_1", RuntimeError("cannot load")]
        db_manager.get_user_for_imu.return_value = UserIdentifier("user_1")
        db_manager.load_user.return_value = "user_data_1"
        db_manager.get_instrument_spec_for_imu.return_value = None
        db_manager.delete_features_by_ids.return_value = [SimpleNamespace(value="feature_1")]

        extractor = MagicMock()
        extractor.extract_gait_features.return_value = "gait_results_1"
        builder = MagicMock()
        builder.build.return_value = SimpleNamespace(
            feature_metadata=SimpleNamespace(feature_identifier=SimpleNamespace(value="feature_1"))
        )

        orchestrator = GaitFeatureDatasetOrchestrator(
            db_manager=db_manager,
            extractor=extractor,
            record_feature_builder=builder,
        )
        orchestrator._write_last_run_summary = MagicMock()  # avoid filesystem writes
        summary = orchestrator.generate_for_all_imu(
            continue_on_error=True,
            rollback_on_partial_failure=True,
        )

        self.assertTrue(summary.rollback_performed)
        self.assertEqual(summary.status, "partial_success_rolled_back")
        db_manager.delete_features_by_ids.assert_called_once()

    def test_all_nan_stride_failure_is_recorded_and_categorized(self):
        db_manager = MagicMock()
        imu_ids = [IMUDataIdentifier("imu_good"), IMUDataIdentifier("imu_nan_stride")]
        db_manager.list_imu_ids.return_value = imu_ids
        db_manager.get_user_for_imu.return_value = UserIdentifier("user_1")
        db_manager.load_imu.side_effect = ["imu_good_data", "imu_nan_data"]
        db_manager.load_user.return_value = "user_data_1"
        db_manager.get_instrument_spec_for_imu.return_value = None

        extractor = MagicMock()
        extractor.extract_gait_features.side_effect = ["gait_good", "gait_nan"]
        builder = MagicMock()
        stride_error = StrideFeatureGenerationError(
            "Stride feature generation failed [skdh_missing_all_stride_keys]: "
            "SKDH output contained none of the expected stride feature keys.",
            diagnosis={
                "stage": "skdh_missing_all_stride_keys",
                "missing_stride_keys": ["stride time"],
                "expected_stride_keys": 36,
                "bouts_without_events": [0],
                "tensor_all_nan": True,
            },
        )
        builder.build.side_effect = [
            SimpleNamespace(
                feature_metadata=SimpleNamespace(
                    feature_identifier=SimpleNamespace(value="feature_good")
                )
            ),
            stride_error,
        ]

        orchestrator = GaitFeatureDatasetOrchestrator(
            db_manager=db_manager,
            extractor=extractor,
            record_feature_builder=builder,
        )
        orchestrator._write_last_run_summary = MagicMock()
        summary = orchestrator.generate_for_all_imu(continue_on_error=True)

        self.assertEqual(
            [item.value for item in summary.generated_feature_ids], ["feature_good"]
        )
        self.assertEqual(
            [item.value for item in summary.failed_imu_ids], ["imu_nan_stride"]
        )
        self.assertEqual(
            [item.value for item in summary.stride_failure_imu_ids], ["imu_nan_stride"]
        )
        self.assertEqual(summary.status, "partial_success")
        self.assertIn("imu_nan_stride", summary.errors_by_imu_id)


if __name__ == "__main__":
    unittest.main()
