import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np
from src.data_model.data.user.clinical.clinical_demographic_data import FallerStatus
from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.services.feature_quality_report import (
    FeatureQualityReportBuilder,
    build_participant_metrics,
    feature_level_summary_text,
    participant_level_summary_text,
)
from src.data_visualization.suite.services.visualization_data_service import (
    VisualizationDataService,
)


@dataclass
class _FakeIdentifier:
    value: str


@dataclass
class _FakeClinicalData:
    faller_status: FallerStatus


@dataclass
class _FakeUserWithClinical:
    user_identifier: _FakeIdentifier
    clinical_demographic_data: _FakeClinicalData


class _FakeDBManager:
    def __init__(self):
        self.feature_ids = [_FakeIdentifier("feature_1"), _FakeIdentifier("feature_2")]
        self.feature_records = {
            "feature_1": RecordFeatures(
                epoch_features=BoutFeatures(
                    sample_basis=SampleBasis.EPOCH,
                    features=np.array([[[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]]]),
                    bout_starts=np.array([0.0]),
                    bout_ends=np.array([3.0]),
                    feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
                    sample_starts=np.array([0.0, 1.0, 2.0]),
                    sample_ends=np.array([1.0, 2.0, 3.0]),
                    units=["m/s", "steps/min"],
                ),
                stride_features=BoutFeatures(
                    sample_basis=SampleBasis.STRIDE,
                    features=np.array([[[10.0, 20.0], [40.0, 50.0]]]),
                    bout_starts=np.array([0.0]),
                    bout_ends=np.array([2.0]),
                    feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
                    sample_starts=np.array([0.0, 1.0]),
                    sample_ends=np.array([1.0, 2.0]),
                    units=["m/s", "steps/min"],
                ),
                feature_metadata=FeatureMetadata(
                    feature_identifier=_FakeIdentifier("feature_1"),
                    user_identifier=_FakeIdentifier("user_faller"),
                    imu_data_identifier=_FakeIdentifier("imu_1"),
                ),
            ),
            "feature_2": RecordFeatures(
                epoch_features=BoutFeatures(
                    sample_basis=SampleBasis.EPOCH,
                    features=np.array([[[3.0, 3.5], [7.0, 7.5]]]),
                    bout_starts=np.array([0.0]),
                    bout_ends=np.array([2.0]),
                    feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
                    sample_starts=np.array([0.0, 1.0]),
                    sample_ends=np.array([1.0, 2.0]),
                    units=["m/s", "steps/min"],
                ),
                stride_features=BoutFeatures(
                    sample_basis=SampleBasis.STRIDE,
                    features=np.array([[[30.0], [70.0]]]),
                    bout_starts=np.array([0.0]),
                    bout_ends=np.array([1.0]),
                    feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
                    sample_starts=np.array([0.0]),
                    sample_ends=np.array([1.0]),
                    units=["m/s", "steps/min"],
                ),
                feature_metadata=FeatureMetadata(
                    feature_identifier=_FakeIdentifier("feature_2"),
                    user_identifier=_FakeIdentifier("user_non_faller"),
                    imu_data_identifier=_FakeIdentifier("imu_2"),
                ),
            ),
        }
        self.repository = self._FakeRepository()

    class _FakeRepository:
        def list_record_ids(self, id_type):
            return []

        def get_record_path(self, id_type, record_id):
            return "/tmp"

        def get_targets(self, source_type, source_id, relation_type=None):
            return []

    def list_feature_ids(self):
        return self.feature_ids

    def load_features(self, feature_id):
        return self.feature_records[feature_id.value]

    def load_user(self, user_id):
        return _FakeUserWithClinical(
            user_identifier=user_id,
            clinical_demographic_data=_FakeClinicalData(
                faller_status=(
                    FallerStatus.FALLER
                    if user_id.value == "user_faller"
                    else FallerStatus.NON_FALLER
                )
            ),
        )


class TestFeatureQualityReport(unittest.TestCase):
    def setUp(self) -> None:
        self.service = VisualizationDataService(db_manager=_FakeDBManager())
        self.builder = FeatureQualityReportBuilder(self.service)

    def test_feature_level_report_counts_usable_samples_only(self):
        report = self.builder.build_feature_level(SampleBasis.EPOCH)

        self.assertEqual(report.num_feature_records, 2)
        self.assertEqual(report.num_participants, 2)
        self.assertEqual(report.total_usable_samples, 5)
        self.assertEqual(len(report.metrics), 2)

        gait_metric = next(
            metric for metric in report.metrics if metric.feature_type == FeatureType.GAIT_SPEED
        )
        self.assertEqual(gait_metric.valid_count, 4)
        self.assertAlmostEqual(gait_metric.valid_fraction, 4.0 / 5.0)

        cadence_metric = next(
            metric for metric in report.metrics if metric.feature_type == FeatureType.CADENCE
        )
        self.assertEqual(cadence_metric.valid_count, 5)
        self.assertAlmostEqual(cadence_metric.missing_fraction, 0.0)

        self.assertIn(FeatureType.GAIT_SPEED, report.separation)
        self.assertEqual(report.correlation_matrix.shape, (2, 2))

    def test_participant_level_report_verifies_usable_counts(self):
        report = self.builder.build_participant_level(SampleBasis.EPOCH)

        self.assertEqual(len(report.participants), 2)
        self.assertEqual(report.inconsistent_count_records, [])

        faller = next(
            metric for metric in report.participants if metric.participant_id == "user_faller"
        )
        self.assertEqual(faller.usable_sample_count, 3)
        self.assertEqual(faller.tensor_slot_count, 3)
        self.assertEqual(faller.padded_slot_count, 0)
        self.assertTrue(faller.count_consistent)
        self.assertAlmostEqual(faller.mean_coverage, (2.0 / 3.0 + 1.0) / 2.0)

        non_faller = next(
            metric
            for metric in report.participants
            if metric.participant_id == "user_non_faller"
        )
        self.assertEqual(non_faller.usable_sample_count, 2)

    def test_participant_metrics_detect_count_mismatch(self):
        coverage = self.service.collect_per_record_coverage(SampleBasis.STRIDE)
        coverage.usable_sample_counts[0] = 999
        metrics = build_participant_metrics(
            coverage, SampleBasis.STRIDE, self.service
        )
        self.assertFalse(metrics[0].count_consistent)
        self.assertEqual(metrics[0].usable_sample_count, 2)

    def test_summary_text_mentions_usable_samples(self):
        feature_report = self.builder.build_feature_level(SampleBasis.STRIDE)
        participant_report = self.builder.build_participant_level(SampleBasis.STRIDE)

        feature_text = feature_level_summary_text(feature_report)
        participant_text = participant_level_summary_text(participant_report)

        self.assertIn("Usable samples (padding excluded): 3", feature_text)
        self.assertIn("stored tensor and coverage scan agree", participant_text)
        self.assertIn("Total usable samples: 3", participant_text)


if __name__ == "__main__":
    unittest.main()
