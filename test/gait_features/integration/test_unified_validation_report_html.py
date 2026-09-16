import unittest
from datetime import datetime, timezone

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.services.feature_quality_report import (
    FeatureLevelMetric,
    FeatureLevelQualityReport,
    ParticipantLevelMetric,
    ParticipantLevelQualityReport,
)
from test.gait_features.integration.unified_validation_report import (
    RecordValidationSnapshot,
    UnifiedValidationReport,
    render_unified_validation_html,
)
from test.gait_features.validation.physiological_validators import PhysiologicalAuditResult
from test.gait_features.validation.reference_validators import GaitSpeedReferenceAudit


class TestUnifiedValidationReportHtml(unittest.TestCase):
    def test_render_includes_quality_and_validation_sections(self):
        report = UnifiedValidationReport(
            generated_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
            dataset="lab_walks",
            extraction_profile="treadmill",
            extraction_backend="skdh",
            feature_level_epoch=FeatureLevelQualityReport(
                basis=SampleBasis.EPOCH,
                feature_types=[FeatureType.GAIT_SPEED],
                metrics=[
                    FeatureLevelMetric(
                        feature_type=FeatureType.GAIT_SPEED,
                        valid_count=8,
                        total_usable_samples=10,
                        valid_fraction=0.8,
                        missing_fraction=0.2,
                    )
                ],
                correlation_matrix=np.eye(1),
                separation={FeatureType.GAIT_SPEED: 0.5},
                classes=["non-faller"],
                total_usable_samples=10,
                num_participants=1,
                num_feature_records=1,
            ),
            feature_level_stride=FeatureLevelQualityReport(
                basis=SampleBasis.STRIDE,
                feature_types=[FeatureType.STRIDE_TIME],
                metrics=[
                    FeatureLevelMetric(
                        feature_type=FeatureType.STRIDE_TIME,
                        valid_count=10,
                        total_usable_samples=10,
                        valid_fraction=1.0,
                        missing_fraction=0.0,
                    )
                ],
                correlation_matrix=np.eye(1),
                separation={FeatureType.STRIDE_TIME: 0.2},
                classes=["non-faller"],
                total_usable_samples=10,
                num_participants=1,
                num_feature_records=1,
            ),
            participant_level_epoch=ParticipantLevelQualityReport(
                basis=SampleBasis.EPOCH,
                feature_types=[FeatureType.GAIT_SPEED],
                participants=[
                    ParticipantLevelMetric(
                        participant_id="CO001",
                        class_label="non-faller",
                        feature_id="feature_1",
                        usable_sample_count=10,
                        tensor_slot_count=10,
                        padded_slot_count=0,
                        mean_coverage=0.95,
                        per_feature_fractions=np.array([0.95]),
                        count_consistent=True,
                    )
                ],
                counts_by_class={"non-faller": np.array([10.0])},
                inconsistent_count_records=[],
                num_feature_records=1,
            ),
            participant_level_stride=ParticipantLevelQualityReport(
                basis=SampleBasis.STRIDE,
                feature_types=[FeatureType.STRIDE_TIME],
                participants=[
                    ParticipantLevelMetric(
                        participant_id="CO001",
                        class_label="non-faller",
                        feature_id="feature_1",
                        usable_sample_count=10,
                        tensor_slot_count=10,
                        padded_slot_count=0,
                        mean_coverage=1.0,
                        per_feature_fractions=np.array([1.0]),
                        count_consistent=True,
                    )
                ],
                counts_by_class={"non-faller": np.array([10.0])},
                inconsistent_count_records=[],
                num_feature_records=1,
            ),
            record_snapshots=[
                RecordValidationSnapshot(
                    participant_id="CO001",
                    feature_id="feature_1",
                    class_label="non-faller",
                    stride_physiological=PhysiologicalAuditResult(basis="stride", passed=True),
                    epoch_physiological=PhysiologicalAuditResult(basis="epoch", passed=True),
                    gait_speed_reference=GaitSpeedReferenceAudit(
                        participant_id="CO001",
                        passed=True,
                        extracted_mps=1.40,
                        reference_mps=1.24,
                        relative_error=0.129,
                        tolerance_relative=0.25,
                    ),
                )
            ],
        )

        html_output = render_unified_validation_html(report)
        self.assertIn("Feature quality — stride basis", html_output)
        self.assertIn("Physiological plausibility", html_output)
        self.assertIn("Gait speed reference", html_output)
        self.assertIn("ALL CHECKS PASSED", html_output)
        self.assertIn("CO001", html_output)


if __name__ == "__main__":
    unittest.main()
