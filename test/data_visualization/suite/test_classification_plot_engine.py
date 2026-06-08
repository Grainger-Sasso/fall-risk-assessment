import unittest

from matplotlib.figure import Figure

from src.classification.evaluation.evaluation_artifact import (
    EvaluationArtifact,
    ModelFamilyResult,
)
from src.classification.evaluation.evaluation_mode import EARLY_FUSION_RESULT_KEY
from src.data_visualization.suite.plots.classification_plot_engine import (
    ClassificationPlotEngine,
)


def _artifact() -> EvaluationArtifact:
    result = ModelFamilyResult(
        model_name="logistic_regression_elastic_net",
        base_results={
            "stride": {"roc_auc_mean": 0.7, "roc_auc_std": 0.05},
            "epoch": {"roc_auc_mean": 0.65, "roc_auc_std": 0.04},
        },
        fusion_results={
            "mean_probability": {
                "roc_auc_mean": 0.75,
                "roc_auc_std": 0.05,
                "pr_auc_mean": 0.7,
                "pr_auc_std": 0.05,
                "balanced_accuracy_mean": 0.72,
            },
            "stacking": {
                "roc_auc_mean": 0.8,
                "roc_auc_std": 0.04,
                "pr_auc_mean": 0.78,
                "pr_auc_std": 0.04,
                "balanced_accuracy_mean": 0.77,
            },
        },
        roc_curves={
            "mean_probability": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.7, 1.0]},
            "stacking": {"fpr": [0.0, 0.3, 1.0], "tpr": [0.0, 0.8, 1.0]},
        },
        pr_curves={
            "mean_probability": {"recall": [1.0, 0.5, 0.0], "precision": [0.5, 0.7, 1.0]},
            "stacking": {"recall": [1.0, 0.6, 0.0], "precision": [0.5, 0.8, 1.0]},
        },
        confusion={
            "mean_probability": {"tn": 5, "fp": 2, "fn": 1, "tp": 6},
            "stacking": {"tn": 6, "fp": 1, "fn": 1, "tp": 6},
        },
    )
    return EvaluationArtifact(
        generated_at="2026-06-04T00:00:00+00:00",
        cv_config={"resolved_n_splits": 2, "n_repeats": 1},
        participant_counts={"common": 8, "faller": 4, "non_faller": 4},
        sample_counts={"stride": 24, "epoch": 32},
        fusion_strategies=["mean_probability", "stacking"],
        model_results=[result],
        ranking=[
            {
                "model_name": "logistic_regression_elastic_net",
                "fusion": "stacking",
                "roc_auc_mean": 0.8,
                "pr_auc_mean": 0.78,
                "balanced_accuracy_mean": 0.77,
                "score": 0.79,
            }
        ],
    )


class TestClassificationPlotEngine(unittest.TestCase):
    def setUp(self):
        self.artifact = _artifact()
        self.model_name = "logistic_regression_elastic_net"

    def _engine(self):
        return ClassificationPlotEngine(Figure(figsize=(8, 6)))

    def test_model_comparison_renders(self):
        engine = self._engine()
        engine.render_model_comparison(self.artifact, metric="roc_auc")
        self.assertTrue(engine.figure.axes)

    def test_roc_and_pr_overlays_render(self):
        engine = self._engine()
        engine.render_roc_overlay(self.artifact, self.model_name)
        self.assertTrue(engine.figure.axes)
        engine.render_pr_overlay(self.artifact, self.model_name)
        self.assertTrue(engine.figure.axes)

    def test_fusion_comparison_and_confusion_render(self):
        engine = self._engine()
        engine.render_fusion_comparison(self.artifact, self.model_name)
        self.assertTrue(engine.figure.axes)
        engine.render_confusion(self.artifact, self.model_name)
        self.assertTrue(engine.figure.axes)

    def test_ranking_table_renders(self):
        engine = self._engine()
        engine.render_ranking_table(self.artifact)
        self.assertTrue(engine.figure.axes)

    def test_best_fusion_selection(self):
        engine = self._engine()
        best = engine.best_fusion(self.artifact.model_results[0], self.artifact)
        self.assertEqual(best, "stacking")

    def test_early_fusion_artifact_renders(self):
        result = ModelFamilyResult(
            model_name="random_forest",
            fusion_results={
                EARLY_FUSION_RESULT_KEY: {
                    "roc_auc_mean": 0.71,
                    "roc_auc_std": 0.04,
                    "pr_auc_mean": 0.69,
                    "pr_auc_std": 0.03,
                    "balanced_accuracy_mean": 0.66,
                }
            },
            roc_curves={
                EARLY_FUSION_RESULT_KEY: {
                    "fpr": [0.0, 0.5, 1.0],
                    "tpr": [0.0, 0.8, 1.0],
                }
            },
            pr_curves={
                EARLY_FUSION_RESULT_KEY: {
                    "recall": [1.0, 0.5, 0.0],
                    "precision": [0.5, 0.8, 1.0],
                }
            },
            confusion={EARLY_FUSION_RESULT_KEY: {"tn": 4, "fp": 1, "fn": 1, "tp": 5}},
        )
        artifact = EvaluationArtifact(
            generated_at="2026-06-05T00:00:00+00:00",
            evaluation_mode="early_fusion_participant",
            aggregation="mean",
            cv_config={"resolved_n_splits": 2, "n_repeats": 1},
            participant_counts={"common": 8, "faller": 4, "non_faller": 4},
            sample_counts={"stride": 24, "epoch": 32},
            fusion_strategies=[EARLY_FUSION_RESULT_KEY],
            model_results=[result],
            ranking=[
                {
                    "model_name": "random_forest",
                    "fusion": EARLY_FUSION_RESULT_KEY,
                    "roc_auc_mean": 0.71,
                    "pr_auc_mean": 0.69,
                    "balanced_accuracy_mean": 0.66,
                    "score": 0.7,
                }
            ],
        )
        engine = self._engine()
        engine.render_model_comparison(artifact, metric="roc_auc")
        engine.render_fusion_comparison(artifact, "random_forest")
        engine.render_ranking_table(artifact)
        engine.render_confusion(artifact, "random_forest")
        self.assertTrue(engine.figure.axes)


if __name__ == "__main__":
    unittest.main()
