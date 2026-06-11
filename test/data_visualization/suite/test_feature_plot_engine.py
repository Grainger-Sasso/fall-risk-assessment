import unittest

import numpy as np
from matplotlib.figure import Figure

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.plots.feature_plot_engine import FeaturePlotEngine
from src.data_visualization.suite.services.feature_quality_report import (
    FeatureLevelMetric,
    FeatureLevelQualityReport,
    ParticipantLevelMetric,
    ParticipantLevelQualityReport,
)
from src.data_visualization.suite.services.visualization_data_service import (
    PerRecordCoverageMatrix,
)


class TestFeaturePlotEngine(unittest.TestCase):
    def test_render_class_violin_returns_summary(self):
        figure = Figure(figsize=(8, 6))
        engine = FeaturePlotEngine(figure)
        grouped = {
            "faller": np.array([1.0, 2.0, 3.0]),
            "non-faller": np.array([2.0, 2.5, 3.5, 4.0]),
        }

        summary = engine.render_class_violin(
            grouped_values=grouped,
            feature_type=FeatureType.GAIT_SPEED,
            basis=SampleBasis.EPOCH,
        )

        self.assertIn("faller", summary)
        self.assertIn("non-faller", summary)
        self.assertEqual(summary["faller"]["count"], 3.0)
        self.assertGreater(summary["non-faller"]["mean"], summary["faller"]["mean"])

    def test_render_sample_count_by_basis_class(self):
        figure = Figure(figsize=(8, 6))
        engine = FeaturePlotEngine(figure)
        counts = {
            "epoch": {
                "faller": np.array([6.0, 8.0]),
                "non-faller": np.array([10.0, 10.0, 10.0]),
            },
            "stride": {
                "faller": np.array([80.0, 100.0]),
                "non-faller": np.array([70.0, 90.0, 110.0]),
            },
        }

        summary = engine.render_sample_count_by_basis_class(counts)

        self.assertEqual(summary["epoch|faller"]["mean"], 7.0)
        self.assertEqual(summary["epoch|faller"]["std"], 1.0)
        self.assertEqual(summary["epoch|non-faller"]["mean"], 10.0)
        self.assertEqual(summary["stride|faller"]["mean"], 90.0)
        self.assertEqual(int(summary["stride|non-faller"]["n_participants"]), 3)
        # Four logical bars: 2 bases x 2 classes.
        self.assertEqual(len(summary), 4)
        self.assertTrue(figure.axes)

    def test_render_sample_count_empty(self):
        figure = Figure(figsize=(8, 6))
        engine = FeaturePlotEngine(figure)
        self.assertEqual(engine.render_sample_count_by_basis_class({}), {})

    def test_compute_correlation_matrix_handles_nan_and_constant(self):
        matrix = np.array(
            [
                [1.0, 2.0, 5.0],
                [2.0, 4.0, 5.0],
                [3.0, 6.0, 5.0],
                [4.0, np.nan, 5.0],
            ]
        )
        correlation = FeaturePlotEngine.compute_correlation_matrix(matrix)
        self.assertEqual(correlation.shape, (3, 3))
        # Columns 0 and 1 are perfectly correlated on complete rows.
        self.assertAlmostEqual(correlation[0, 1], 1.0, places=6)
        # Constant column 2 has no defined correlation.
        self.assertTrue(np.isnan(correlation[0, 2]))

    def test_render_correlation_heatmap_returns_matrix(self):
        figure = Figure(figsize=(6, 6))
        engine = FeaturePlotEngine(figure)
        matrix = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        correlation = engine.render_feature_correlation_heatmap(
            feature_types=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
            feature_matrix=matrix,
            basis=SampleBasis.EPOCH,
        )
        self.assertEqual(correlation.shape, (2, 2))

    def test_compute_class_separation_two_classes(self):
        matrix = np.array(
            [
                [1.0, 10.0],
                [1.2, 10.0],
                [0.9, 10.0],
                [5.0, 10.0],
                [5.2, 10.0],
                [4.8, 10.0],
            ]
        )
        labels = ["faller", "faller", "faller", "non-faller", "non-faller", "non-faller"]
        separation, classes = FeaturePlotEngine.compute_class_separation(
            matrix, labels, [FeatureType.GAIT_SPEED, FeatureType.CADENCE]
        )
        self.assertEqual(classes, ["faller", "non-faller"])
        self.assertIn(FeatureType.GAIT_SPEED, separation)
        # Constant feature across classes has zero pooled separation signal.
        self.assertNotIn(FeatureType.CADENCE, separation)
        self.assertLess(separation[FeatureType.GAIT_SPEED], 0.0)

    def test_compute_feature_coverage_reports_empty_feature(self):
        matrix = np.array(
            [
                [1.0, np.nan],
                [2.0, np.nan],
                [np.nan, np.nan],
            ]
        )
        labels = ["faller", "faller", "non-faller"]
        coverage = FeaturePlotEngine.compute_feature_coverage(
            matrix, labels, [FeatureType.GAIT_SPEED, FeatureType.CADENCE]
        )
        self.assertAlmostEqual(coverage[FeatureType.GAIT_SPEED]["valid_fraction"], 2.0 / 3.0)
        self.assertEqual(coverage[FeatureType.CADENCE]["valid_fraction"], 0.0)
        self.assertEqual(
            coverage[FeatureType.GAIT_SPEED]["per_class"]["faller"]["valid"], 2
        )

    def test_render_per_record_missingness_heatmap(self):
        figure = Figure(figsize=(8, 6))
        engine = FeaturePlotEngine(figure)
        coverage = PerRecordCoverageMatrix(
            feature_types=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
            matrix=np.array(
                [
                    [1.0, 0.0],
                    [0.5, 1.0],
                ]
            ),
            feature_ids=["feat-a", "feat-b"],
            participant_ids=["u1", "u2"],
            class_labels=["faller", "non-faller"],
            usable_sample_counts=[10, 20],
        )

        summary = engine.render_per_record_missingness_heatmap(
            coverage=coverage,
            basis=SampleBasis.EPOCH,
            sort_by="worst_first",
            row_label_mode="participant",
        )

        self.assertEqual(summary["n_records"], 2)
        self.assertEqual(summary["records_with_any_missing"], 2)
        self.assertEqual(summary["records_with_fully_missing_feature"], 1)
        self.assertTrue(figure.axes)


    def test_render_feature_missingness_from_report(self):
        figure = Figure(figsize=(8, 6))
        engine = FeaturePlotEngine(figure)
        report = FeatureLevelQualityReport(
            basis=SampleBasis.EPOCH,
            feature_types=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
            metrics=[
                FeatureLevelMetric(
                    feature_type=FeatureType.GAIT_SPEED,
                    valid_count=8,
                    total_usable_samples=10,
                    valid_fraction=0.8,
                    missing_fraction=0.2,
                ),
                FeatureLevelMetric(
                    feature_type=FeatureType.CADENCE,
                    valid_count=10,
                    total_usable_samples=10,
                    valid_fraction=1.0,
                    missing_fraction=0.0,
                ),
            ],
            correlation_matrix=np.eye(2),
            separation={FeatureType.GAIT_SPEED: -1.2},
            classes=["faller", "non-faller"],
            total_usable_samples=10,
            num_participants=2,
            num_feature_records=2,
        )

        missingness = engine.render_feature_missingness(report)
        self.assertAlmostEqual(missingness[FeatureType.GAIT_SPEED], 0.2)
        self.assertTrue(figure.axes)

    def test_render_participant_sample_counts(self):
        figure = Figure(figsize=(8, 6))
        engine = FeaturePlotEngine(figure)
        report = ParticipantLevelQualityReport(
            basis=SampleBasis.STRIDE,
            feature_types=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
            participants=[
                ParticipantLevelMetric(
                    participant_id="u1",
                    class_label="faller",
                    feature_id="f1",
                    usable_sample_count=80,
                    tensor_slot_count=100,
                    padded_slot_count=20,
                    mean_coverage=0.9,
                    per_feature_fractions=np.array([1.0, 0.8]),
                    count_consistent=True,
                ),
                ParticipantLevelMetric(
                    participant_id="u2",
                    class_label="non-faller",
                    feature_id="f2",
                    usable_sample_count=120,
                    tensor_slot_count=120,
                    padded_slot_count=0,
                    mean_coverage=1.0,
                    per_feature_fractions=np.array([1.0, 1.0]),
                    count_consistent=True,
                ),
            ],
            counts_by_class={"faller": np.array([80.0]), "non-faller": np.array([120.0])},
            inconsistent_count_records=[],
            num_feature_records=2,
        )

        counts = engine.render_participant_sample_counts(report, sort_by="count_asc")
        self.assertEqual(counts["u1"], 80)
        self.assertEqual(counts["u2"], 120)
        self.assertTrue(figure.axes)

    def test_render_participant_coverage_heatmap(self):
        figure = Figure(figsize=(8, 6))
        engine = FeaturePlotEngine(figure)
        report = ParticipantLevelQualityReport(
            basis=SampleBasis.EPOCH,
            feature_types=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
            participants=[
                ParticipantLevelMetric(
                    participant_id="u1",
                    class_label="faller",
                    feature_id="f1",
                    usable_sample_count=10,
                    tensor_slot_count=10,
                    padded_slot_count=0,
                    mean_coverage=0.5,
                    per_feature_fractions=np.array([1.0, 0.0]),
                    count_consistent=True,
                )
            ],
            counts_by_class={"faller": np.array([10.0])},
            inconsistent_count_records=[],
            num_feature_records=1,
        )

        summary = engine.render_participant_coverage_heatmap(report)
        self.assertEqual(summary["n_participants"], 1)
        self.assertEqual(summary["total_usable_samples"], 10)
        self.assertEqual(summary["participants_with_any_missing"], 1)


if __name__ == "__main__":
    unittest.main()
