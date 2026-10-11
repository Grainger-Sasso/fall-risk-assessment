import os
import unittest
from pathlib import Path

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.services.feature_quality_report import FeatureQualityReportBuilder
from src.data_visualization.suite.services.visualization_data_service import VisualizationDataService
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.gait_feature_dataset_orchestrator import GaitFeatureDatasetOrchestrator
from test.fixtures.assessment_database import (
    imu_identifier,
    participant_fixtures,
    temp_database_manager,
)
from test.fixtures.integration_cases import SkdhIntegrationTestCase
from test.gait_features.integration.unified_validation_report import (
    DEFAULT_ARTIFACT_PATH,
    build_unified_validation_report,
    write_unified_validation_artifact,
)


class TestUnifiedValidationReportIntegration(SkdhIntegrationTestCase):
    """Generate a unified HTML artifact covering quality reports and validation audits."""

    def test_unified_validation_report_for_lab_walks(self):
        fixtures = participant_fixtures("lab_walks", tier="primary")[:2]
        imu_ids = [imu_identifier(item.imu_id) for item in fixtures]
        fixture_by_imu = {item.imu_id: item for item in fixtures}

        with temp_database_manager("lab_walks") as context:
            orchestrator = GaitFeatureDatasetOrchestrator(
                db_manager=context.db_manager,
                treadmill_profile=True,
                extraction_backend=GaitExtractionBackendId.SKDH,
                extraction_profile=ExtractionProfile.TREADMILL,
            )
            orchestrator._write_last_run_summary = lambda summary: None  # noqa: SLF001
            summary = orchestrator.generate_for_imu_ids(imu_ids=imu_ids)
            self.assertEqual(summary.status, "success")
            self.assertEqual(len(summary.generated_feature_ids), len(imu_ids))

            data_service = VisualizationDataService(db_manager=context.db_manager)
            report_builder = FeatureQualityReportBuilder(data_service=data_service)

            for basis in (SampleBasis.EPOCH, SampleBasis.STRIDE):
                with self.subTest(basis=basis.value):
                    feature_report = report_builder.build_feature_level(basis)
                    self.assertFalse(feature_report.is_empty)
                    self.assertEqual(feature_report.num_feature_records, len(imu_ids))

            records = []
            for feature_id in summary.generated_feature_ids:
                record = context.db_manager.load_features(feature_id)
                imu_key = record.feature_metadata.imu_data_identifier.value
                fixture = fixture_by_imu[imu_key]
                records.append((fixture.participant_id, feature_id.value, record))

            unified = build_unified_validation_report(
                dataset="lab_walks",
                extraction_profile=ExtractionProfile.TREADMILL.value,
                extraction_backend=GaitExtractionBackendId.SKDH.value,
                data_service=data_service,
                records=records,
            )
            self.assertTrue(unified.all_checks_passed)
            self.assertEqual(len(unified.record_snapshots), len(imu_ids))
            for snapshot in unified.record_snapshots:
                self.assertTrue(snapshot.stride_physiological.passed)
                self.assertTrue(snapshot.epoch_physiological.passed)
                self.assertIsNotNone(snapshot.gait_speed_reference)
                self.assertTrue(snapshot.gait_speed_reference.passed)

            if os.environ.get("GENERATE_GAIT_VALIDATION_ARTIFACTS", "").strip():
                artifact_path = write_unified_validation_artifact(unified, DEFAULT_ARTIFACT_PATH)
                self.assertTrue(artifact_path.is_file())
                content = artifact_path.read_text(encoding="utf-8")
                self.assertIn("Gait Feature Validation Report", content)
                self.assertIn("Physiological plausibility", content)
                self.assertIn("Gait speed reference", content)
                self.assertIn("Feature quality — stride basis", content)


if __name__ == "__main__":
    unittest.main()
