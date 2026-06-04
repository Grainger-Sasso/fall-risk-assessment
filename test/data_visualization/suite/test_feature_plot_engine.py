import unittest

import numpy as np
from matplotlib.figure import Figure

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.plots.feature_plot_engine import FeaturePlotEngine


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


if __name__ == "__main__":
    unittest.main()
