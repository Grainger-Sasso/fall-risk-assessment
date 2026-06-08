import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.classification.data.classification_dataset import EarlyFusionParticipantDataset
from src.classification.evaluation.evaluation_artifact import EvaluationArtifact
from src.classification.evaluation.evaluation_mode import (
    EARLY_FUSION_PARTICIPANT,
    EARLY_FUSION_RESULT_KEY,
)
from src.classification.evaluation.participant_early_fusion_evaluator import (
    ParticipantEarlyFusionEvaluator,
)
from src.data_types.feature.feature_type import FeatureType


def _make_dataset(seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    participant_ids = []
    for label in (0, 1):
        for index in range(4):
            participant = f"{'F' if label == 1 else 'N'}{index}"
            center = 1.0 if label == 1 else 0.0
            rows.append(rng.normal(center, 0.2, size=8))
            labels.append(label)
            participant_ids.append(participant)
    return EarlyFusionParticipantDataset(
        X=np.vstack(rows),
        y=np.asarray(labels, dtype=int),
        participant_ids=participant_ids,
        stride_feature_names=[
            FeatureType.GAIT_SPEED,
            FeatureType.CADENCE,
            FeatureType.STRIDE_TIME,
            FeatureType.STEP_TIME,
        ],
        epoch_feature_names=[
            FeatureType.EPOCH_VERTICAL_MEAN,
            FeatureType.EPOCH_VERTICAL_STD,
            FeatureType.EPOCH_MEDIOLATERAL_MEAN,
            FeatureType.EPOCH_MEDIOLATERAL_STD,
        ],
        aggregation="mean",
    )


class TestParticipantEarlyFusionEvaluator(unittest.TestCase):
    def test_end_to_end_and_artifact_round_trip(self):
        dataset = _make_dataset()
        evaluator = ParticipantEarlyFusionEvaluator(
            dataset=dataset,
            sample_counts={"stride": 24, "epoch": 32},
            model_names=["logistic_regression_elastic_net"],
            n_splits=2,
            n_repeats=1,
            fast_mode=True,
            verbose=False,
        )
        artifact = evaluator.evaluate()

        self.assertEqual(artifact.evaluation_mode, EARLY_FUSION_PARTICIPANT)
        self.assertEqual(artifact.aggregation, "mean")
        self.assertEqual(artifact.fusion_strategies, [EARLY_FUSION_RESULT_KEY])
        self.assertEqual(len(artifact.model_results), 1)
        result = artifact.model_results[0]
        self.assertEqual(result.model_name, "logistic_regression_elastic_net")
        self.assertEqual(result.base_results, {})
        self.assertIn(EARLY_FUSION_RESULT_KEY, result.fusion_results)
        self.assertIn("roc_auc_mean", result.fusion_results[EARLY_FUSION_RESULT_KEY])
        self.assertTrue(artifact.ranking)
        self.assertEqual(artifact.ranking[0]["fusion"], EARLY_FUSION_RESULT_KEY)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "artifact.json"
            artifact.to_json(path)
            reloaded = EvaluationArtifact.from_json(path)

        self.assertEqual(reloaded.evaluation_mode, EARLY_FUSION_PARTICIPANT)
        self.assertEqual(reloaded.aggregation, "mean")
        self.assertEqual(
            reloaded.model_results[0].fusion_results[EARLY_FUSION_RESULT_KEY][
                "roc_auc_mean"
            ],
            result.fusion_results[EARLY_FUSION_RESULT_KEY]["roc_auc_mean"],
        )


if __name__ == "__main__":
    unittest.main()
