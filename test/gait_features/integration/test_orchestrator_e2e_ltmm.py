import unittest

from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.gait_feature_dataset_orchestrator import GaitFeatureDatasetOrchestrator
from test.gait_features.integration.base import LtmmIntegrationTestCase
from test.gait_features.fixtures.fixture_loader import (
    imu_identifier,
    participant_fixtures,
    temp_database_manager,
)
from test.gait_features.validation.extraction_validators import (
    assert_epoch_features_reliable,
    assert_record_features_schema,
)
from test.gait_features.validation.physiological_validators import (
    assert_epoch_physiological_plausibility,
    assert_stride_physiological_plausibility,
)


class TestOrchestratorE2eLtmm(LtmmIntegrationTestCase):
    """DatabaseManager + orchestrator on whole LTMM files (isolated temp index copy)."""

    def test_orchestrator_generates_features_for_primary_ltmm_record(self):
        fixture = participant_fixtures("ltmm", tier="primary")[0]
        imu_id = imu_identifier(fixture.imu_id)

        with temp_database_manager("ltmm") as context:
            orchestrator = GaitFeatureDatasetOrchestrator(
                db_manager=context.db_manager,
                extraction_backend=GaitExtractionBackendId.SKDH,
                extraction_profile=ExtractionProfile.FREE_LIVING,
            )
            orchestrator._write_last_run_summary = lambda summary: None  # noqa: SLF001
            summary = orchestrator.generate_for_imu_ids(imu_ids=[imu_id])

            self.assertEqual(summary.status, "success")
            self.assertEqual(len(summary.generated_feature_ids), 1)
            self.assertEqual(summary.failed_imu_ids, [])

            record = context.db_manager.load_features(summary.generated_feature_ids[0])
            assert_record_features_schema(record)
            assert_epoch_features_reliable(record)
            assert_stride_physiological_plausibility(record)
            assert_epoch_physiological_plausibility(record)
            self.assertGreater(record.epoch_features.num_bouts, 0)
            self.assertGreater(record.stride_features.usable_sample_count, 0)
            self.assertEqual(
                record.feature_metadata.imu_data_identifier.value,
                fixture.imu_id,
            )
            self.assertEqual(
                record.feature_metadata.extraction_profile,
                ExtractionProfile.FREE_LIVING.value,
            )

            user_id = context.db_manager.get_user_for_imu(imu_id)
            self.assertIsNotNone(user_id)
            self.assertEqual(user_id.value, fixture.participant_id)


if __name__ == "__main__":
    unittest.main()
