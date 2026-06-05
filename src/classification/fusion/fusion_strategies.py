"""Late-fusion strategies combining stride and epoch participant scores.

Every strategy consumes a per-participant base-score matrix of shape
``(n_participants, n_bases)`` where columns are the pooled stride and epoch
probabilities, and produces a single fused probability per participant.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

MEAN_PROBABILITY = "mean_probability"
WEIGHTED_PROBABILITY = "weighted_probability"
VOTING = "voting"
STACKING = "stacking"


def _clean(base_scores: np.ndarray) -> np.ndarray:
    """Replace NaN base scores with the neutral probability 0.5."""
    scores = np.asarray(base_scores, dtype=float)
    return np.where(np.isfinite(scores), scores, 0.5)


class FusionStrategy(ABC):
    name: str
    requires_training: bool = False

    @abstractmethod
    def fit(self, base_scores: np.ndarray, y: np.ndarray) -> "FusionStrategy":
        """Fit any internal parameters (no-op for heuristic strategies)."""

    @abstractmethod
    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        """Return a fused probability per participant."""


class MeanProbabilityFusion(FusionStrategy):
    name = MEAN_PROBABILITY
    requires_training = False

    def fit(self, base_scores: np.ndarray, y: np.ndarray) -> "MeanProbabilityFusion":
        return self

    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        return _clean(base_scores).mean(axis=1)


class WeightedProbabilityFusion(FusionStrategy):
    """Convex combination of base scores with a weight learned on training data."""

    name = WEIGHTED_PROBABILITY
    requires_training = True

    def __init__(self, weight_grid: List[float] = None) -> None:
        self.weight_grid = weight_grid or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.weights_: np.ndarray = np.array([0.5, 0.5])

    def fit(
        self, base_scores: np.ndarray, y: np.ndarray
    ) -> "WeightedProbabilityFusion":
        from sklearn.metrics import roc_auc_score

        scores = _clean(base_scores)
        y = np.asarray(y).astype(int)
        if scores.shape[1] != 2 or len(np.unique(y)) < 2:
            self.weights_ = np.array([0.5, 0.5])
            return self

        best_weight = 0.5
        best_auc = -1.0
        for stride_weight in self.weight_grid:
            fused = stride_weight * scores[:, 0] + (1.0 - stride_weight) * scores[:, 1]
            try:
                auc = roc_auc_score(y, fused)
            except ValueError:
                continue
            if auc > best_auc:
                best_auc = auc
                best_weight = stride_weight
        self.weights_ = np.array([best_weight, 1.0 - best_weight])
        return self

    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        return _clean(base_scores) @ self.weights_


class VotingFusion(FusionStrategy):
    """Soft vote: fraction of bases that classify the participant as positive."""

    name = VOTING
    requires_training = False

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def fit(self, base_scores: np.ndarray, y: np.ndarray) -> "VotingFusion":
        return self

    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        votes = (_clean(base_scores) >= self.threshold).astype(float)
        return votes.mean(axis=1)


class StackingMetaFusion(FusionStrategy):
    """Logistic-regression meta-classifier over the base scores."""

    name = STACKING
    requires_training = True

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._meta_learner = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        self._fallback_prob = 0.5
        self._is_fitted = False

    def fit(self, base_scores: np.ndarray, y: np.ndarray) -> "StackingMetaFusion":
        scores = _clean(base_scores)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            self._fallback_prob = float(y.mean()) if y.size else 0.5
            self._is_fitted = False
            return self
        self._meta_learner.fit(scores, y)
        self._is_fitted = True
        return self

    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        scores = _clean(base_scores)
        if not self._is_fitted:
            return np.full(scores.shape[0], self._fallback_prob, dtype=float)
        return self._meta_learner.predict_proba(scores)[:, 1]


_FUSION_BUILDERS: Dict[str, type] = {
    MEAN_PROBABILITY: MeanProbabilityFusion,
    WEIGHTED_PROBABILITY: WeightedProbabilityFusion,
    VOTING: VotingFusion,
    STACKING: StackingMetaFusion,
}


def available_fusion_names() -> List[str]:
    return list(_FUSION_BUILDERS.keys())


def create_fusion_strategy(name: str, random_state: int = 42) -> FusionStrategy:
    if name not in _FUSION_BUILDERS:
        raise ValueError(
            f"Unknown fusion strategy '{name}'. Available: {available_fusion_names()}"
        )
    if name == STACKING:
        return StackingMetaFusion(random_state=random_state)
    return _FUSION_BUILDERS[name]()
