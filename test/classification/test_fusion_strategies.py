import unittest

import numpy as np

from src.classification.fusion.fusion_strategies import (
    MeanProbabilityFusion,
    StackingMetaFusion,
    VotingFusion,
    WeightedProbabilityFusion,
    available_fusion_names,
    create_fusion_strategy,
)


class TestFusionStrategies(unittest.TestCase):
    def test_mean_probability(self):
        scores = np.array([[0.2, 0.4], [0.6, 0.8]])
        fused = MeanProbabilityFusion().predict_proba(scores)
        np.testing.assert_allclose(fused, [0.3, 0.7])

    def test_mean_handles_nan_as_neutral(self):
        scores = np.array([[np.nan, 0.4]])
        fused = MeanProbabilityFusion().predict_proba(scores)
        np.testing.assert_allclose(fused, [(0.5 + 0.4) / 2])

    def test_voting_fraction_positive(self):
        scores = np.array([[0.6, 0.6], [0.6, 0.2], [0.1, 0.2]])
        fused = VotingFusion().predict_proba(scores)
        np.testing.assert_allclose(fused, [1.0, 0.5, 0.0])

    def test_weighted_learns_to_favor_informative_base(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Stride is correctly ordered; epoch is actively misleading (anti-correlated),
        # so favoring stride strictly maximizes AUC.
        stride = np.where(y == 1, 0.9, 0.1)
        epoch = np.where(y == 1, 0.1, 0.9)
        scores = np.column_stack([stride, epoch])
        strategy = WeightedProbabilityFusion()
        strategy.fit(scores, y)
        self.assertGreater(strategy.weights_[0], strategy.weights_[1])

    def test_stacking_fits_and_predicts_in_unit_interval(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
        strategy = StackingMetaFusion(random_state=0)
        strategy.fit(scores, y)
        preds = strategy.predict_proba(scores)
        self.assertEqual(preds.shape, (4,))
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))

    def test_stacking_single_class_fallback(self):
        y = np.array([1, 1, 1])
        scores = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]])
        strategy = StackingMetaFusion()
        strategy.fit(scores, y)
        preds = strategy.predict_proba(scores)
        self.assertEqual(preds.shape, (3,))

    def test_factory_round_trip(self):
        for name in available_fusion_names():
            strategy = create_fusion_strategy(name)
            self.assertEqual(strategy.name, name)


if __name__ == "__main__":
    unittest.main()
