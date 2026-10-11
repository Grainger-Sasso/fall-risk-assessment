import unittest

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


class TestFeatureQualityReportIntegration(SkdhIntegrationTestCase):
    """Structured quality reports over real extracted Lab Walks features."""

    def test_quality_reports_summarize_extracted_lab_walks_features(self):
        fixtures = participant_fixtures("lab_walks", tier="primary")[:2]
        imu_ids = [imu_identifier(item.imu_id) for item in fixtures]

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
                    participant_report = report_builder.build_participant_level(basis)
                    self.assertFalse(feature_report.is_empty)
                    self.assertFalse(participant_report.is_empty)
                    self.assertEqual(feature_report.num_feature_records, len(imu_ids))
                    self.assertGreater(feature_report.total_usable_samples, 0)

                    if basis == SampleBasis.STRIDE:
                        stride_metric = next(
                            (
                                metric
                                for metric in feature_report.metrics
                                if metric.feature_type == FeatureType.STRIDE_TIME
                            ),
                            None,
                        )
                        self.assertIsNotNone(stride_metric)
                        self.assertGreater(stride_metric.valid_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
