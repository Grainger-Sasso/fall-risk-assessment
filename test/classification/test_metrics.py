import unittest

import numpy as np

from src.classification.evaluation.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
    compute_roc_curve,
    ranking_score,
)


class TestMetrics(unittest.TestCase):
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_classification_metrics(y_true, y_proba)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["tp"], 2.0)
        self.assertEqual(metrics["tn"], 2.0)

    def test_single_class_yields_nan_auc(self):
        y_true = np.array([1, 1, 1])
        y_proba = np.array([0.6, 0.7, 0.8])
        metrics = compute_classification_metrics(y_true, y_proba)
        self.assertTrue(np.isnan(metrics["roc_auc"]))
        self.assertTrue(np.isnan(metrics["pr_auc"]))

    def test_aggregate_fold_metrics_mean_std(self):
        folds = [{"roc_auc": 0.8}, {"roc_auc": 1.0}]
        summary = aggregate_fold_metrics(folds)
        self.assertAlmostEqual(summary["roc_auc_mean"], 0.9)
        self.assertEqual(summary["n_folds"], 2.0)

    def test_ranking_score_handles_all_nan(self):
        self.assertEqual(ranking_score({"roc_auc_mean": float("nan"), "pr_auc_mean": float("nan")}), -1.0)
        self.assertAlmostEqual(ranking_score({"roc_auc_mean": 0.8, "pr_auc_mean": 0.6}), 0.7)

    def test_roc_curve_empty_for_single_class(self):
        curve = compute_roc_curve(np.array([1, 1]), np.array([0.5, 0.6]))
        self.assertEqual(curve, {"fpr": [], "tpr": []})


if __name__ == "__main__":
    unittest.main()
