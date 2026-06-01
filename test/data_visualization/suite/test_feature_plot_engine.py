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


if __name__ == "__main__":
    unittest.main()
