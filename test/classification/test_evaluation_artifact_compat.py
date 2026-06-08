import tempfile
import unittest
from pathlib import Path

from src.classification.evaluation.evaluation_artifact import (
    EvaluationArtifact,
    ModelFamilyResult,
)
from src.classification.evaluation.evaluation_mode import LATE_FUSION_SAMPLE


class TestEvaluationArtifactCompat(unittest.TestCase):
    def test_legacy_payload_defaults_to_late_fusion_mode(self):
        payload = {
            "generated_at": "2026-06-04T00:00:00+00:00",
            "cv_config": {"resolved_n_splits": 2, "n_repeats": 1},
            "participant_counts": {"common": 8},
            "sample_counts": {"stride": 24, "epoch": 32},
            "fusion_strategies": ["mean_probability"],
            "model_results": [
                {
                    "model_name": "logistic_regression_elastic_net",
                    "base_results": {
                        "stride": {"roc_auc_mean": 0.7},
                        "epoch": {"roc_auc_mean": 0.6},
                    },
                    "fusion_results": {
                        "mean_probability": {"roc_auc_mean": 0.75},
                    },
                    "roc_curves": {},
                    "pr_curves": {},
                    "confusion": {},
                    "best_params": {},
                }
            ],
            "ranking": [],
        }
        artifact = EvaluationArtifact.from_dict(payload)
        self.assertEqual(artifact.evaluation_mode, LATE_FUSION_SAMPLE)
        self.assertIsNone(artifact.aggregation)
        self.assertFalse(artifact.is_early_fusion_participant())

    def test_round_trip_preserves_new_fields(self):
        artifact = EvaluationArtifact(
            generated_at="2026-06-05T00:00:00+00:00",
            evaluation_mode="early_fusion_participant",
            aggregation="mean",
            cv_config={"resolved_n_splits": 2},
            participant_counts={"common": 8},
            sample_counts={"stride": 24, "epoch": 32},
            fusion_strategies=["early_fusion"],
            model_results=[
                ModelFamilyResult(
                    model_name="random_forest",
                    fusion_results={"early_fusion": {"roc_auc_mean": 0.7}},
                )
            ],
            ranking=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "artifact.json"
            artifact.to_json(path)
            reloaded = EvaluationArtifact.from_json(path)
        self.assertTrue(reloaded.is_early_fusion_participant())
        self.assertEqual(reloaded.aggregation, "mean")


if __name__ == "__main__":
    unittest.main()
