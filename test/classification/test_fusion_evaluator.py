import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.classification.data.classification_dataset import (
    BasisSampleDataset,
    ClassificationDataset,
)
from src.classification.evaluation.evaluation_artifact import EvaluationArtifact
from src.classification.evaluation.evaluation_mode import LATE_FUSION_SAMPLE
from src.classification.evaluation.fusion_evaluator import FusionEvaluator
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis


def _make_basis(basis, n_participants_per_class, samples_per_participant, seed):
    rng = np.random.default_rng(seed)
    x_rows = []
    y_rows = []
    groups = []
    for label in (0, 1):
        for participant_index in range(n_participants_per_class):
            participant = f"{'F' if label == 1 else 'N'}{participant_index}"
            center = 1.0 if label == 1 else 0.0
            block = rng.normal(center, 0.3, size=(samples_per_participant, 4))
            x_rows.append(block)
            y_rows.extend([label] * samples_per_participant)
            groups.extend([participant] * samples_per_participant)
    return BasisSampleDataset(
        basis=basis,
        X=np.vstack(x_rows),
        y=np.asarray(y_rows, dtype=int),
        groups=np.asarray(groups, dtype=object),
        feature_names=[
            FeatureType.GAIT_SPEED,
            FeatureType.CADENCE,
            FeatureType.STRIDE_TIME,
            FeatureType.STEP_TIME,
        ],
    )


def _make_dataset(seed=7):
    return ClassificationDataset(
        stride=_make_basis(SampleBasis.STRIDE, 4, 3, seed),
        epoch=_make_basis(SampleBasis.EPOCH, 4, 4, seed + 100),
    )


class TestFusionEvaluator(unittest.TestCase):
    def test_end_to_end_and_artifact_round_trip(self):
        dataset = _make_dataset()
        evaluator = FusionEvaluator(
            dataset=dataset,
            model_names=["logistic_regression_elastic_net"],
            n_splits=2,
            n_repeats=1,
            inner_splits=2,
            fast_mode=True,
            verbose=False,
        )
        artifact = evaluator.evaluate()

        self.assertEqual(len(artifact.model_results), 1)
        result = artifact.model_results[0]
        self.assertEqual(result.model_name, "logistic_regression_elastic_net")
        self.assertIn("stride", result.base_results)
        self.assertIn("epoch", result.base_results)
        for fusion_name in artifact.fusion_strategies:
            self.assertIn(fusion_name, result.fusion_results)
            self.assertIn("roc_auc_mean", result.fusion_results[fusion_name])
        self.assertTrue(artifact.ranking)
        self.assertEqual(artifact.evaluation_mode, LATE_FUSION_SAMPLE)
        self.assertIsNone(artifact.aggregation)
        self.assertEqual(artifact.participant_counts["common"], 8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "artifact.json"
            artifact.to_json(path)
            reloaded = EvaluationArtifact.from_json(path)

        self.assertEqual(reloaded.fusion_strategies, artifact.fusion_strategies)
        self.assertEqual(
            reloaded.model_results[0].model_name, result.model_name
        )
        self.assertEqual(len(reloaded.ranking), len(artifact.ranking))


if __name__ == "__main__":
    unittest.main()
