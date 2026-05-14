from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from src.classification.feature_preprocessor.feature_preprocessor import PreparedDataset


@dataclass
class ModelEvaluationResult:
    model_name: str
    metrics: Dict[str, float]
    best_hyperparameters: Optional[Dict[str, object]] = None


class BaseClassifierModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier for reports."""

    @abstractmethod
    def evaluate(
        self,
        prepared_data: PreparedDataset,
        n_splits: int,
        n_repeats: int,
        random_state: int,
    ) -> ModelEvaluationResult:
        """Run CV evaluation and return aggregated metrics."""
