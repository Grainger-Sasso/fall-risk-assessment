"""Participant-level early-fusion evaluator.

Aggregated stride and epoch features are concatenated into one row per
participant before model training and inference. Cross-validation splits
participants directly (no sample pooling step).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix

from src.classification.data.classification_dataset import EarlyFusionParticipantDataset
from src.classification.evaluation.cross_validation import (
    InsufficientDataError,
    iter_participant_folds,
    resolve_n_splits,
)
from src.classification.evaluation.evaluation_artifact import (
    EvaluationArtifact,
    ModelFamilyResult,
)
from src.classification.evaluation.evaluation_mode import (
    EARLY_FUSION_PARTICIPANT,
    EARLY_FUSION_RESULT_KEY,
)
from src.classification.evaluation.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
    compute_pr_curve,
    compute_roc_curve,
    ranking_score,
)
from src.classification.models.model_factory import available_model_names, create_model


@dataclass
class ParticipantEarlyFusionEvaluator:
    dataset: EarlyFusionParticipantDataset
    sample_counts: Dict[str, int]
    model_names: List[str] = field(default_factory=available_model_names)
    n_splits: int = 5
    n_repeats: int = 5
    inner_splits: int = 3
    random_state: int = 42
    fast_mode: bool = False
    verbose: bool = True

    def evaluate(self) -> EvaluationArtifact:
        if self.dataset.n_participants < 4:
            raise InsufficientDataError(
                "Need at least 4 participants for early-fusion evaluation; "
                f"found {self.dataset.n_participants}."
            )

        participant_labels = np.asarray(self.dataset.y, dtype=int)
        resolved_splits = resolve_n_splits(participant_labels, desired=self.n_splits)

        model_results: List[ModelFamilyResult] = []
        for model_name in self.model_names:
            self._log(f"Evaluating model family: {model_name}")
            model_results.append(
                self._evaluate_family(
                    model_name=model_name,
                    participant_labels=participant_labels,
                    n_splits=resolved_splits,
                )
            )
            self._log(f"  completed {model_name}")

        ranking = self._build_ranking(model_results)
        return EvaluationArtifact(
            generated_at=datetime.now(timezone.utc).isoformat(),
            evaluation_mode=EARLY_FUSION_PARTICIPANT,
            aggregation=self.dataset.aggregation,
            cv_config={
                "requested_n_splits": self.n_splits,
                "resolved_n_splits": resolved_splits,
                "n_repeats": self.n_repeats,
                "inner_splits": self.inner_splits,
                "random_state": self.random_state,
                "fast_mode": self.fast_mode,
            },
            participant_counts={
                "common": self.dataset.n_participants,
                "faller": int(np.sum(participant_labels == 1)),
                "non_faller": int(np.sum(participant_labels == 0)),
                "stride_only": self.dataset.n_participants,
                "epoch_only": self.dataset.n_participants,
            },
            sample_counts=dict(self.sample_counts),
            fusion_strategies=[EARLY_FUSION_RESULT_KEY],
            model_results=model_results,
            ranking=ranking,
        )

    def _evaluate_family(
        self,
        model_name: str,
        participant_labels: np.ndarray,
        n_splits: int,
    ) -> ModelFamilyResult:
        fold_metrics: List[Dict[str, float]] = []
        oof_y: List[int] = []
        oof_p: List[float] = []
        confusion_acc = np.zeros(4, dtype=float)
        best_params: Dict[str, object] = {}

        for train_idx, test_idx in iter_participant_folds(
            participant_labels, n_splits, self.n_repeats, self.random_state
        ):
            model = create_model(
                model_name,
                random_state=self.random_state,
                fast_mode=self.fast_mode,
                enable_tuning=True,
            )
            try:
                model.fit(
                    self.dataset.X[train_idx],
                    participant_labels[train_idx],
                    groups=None,
                )
                scores = model.predict_proba(self.dataset.X[test_idx])
                best_params = dict(getattr(model, "best_params_", {}))
            except Exception:
                scores = np.full(len(test_idx), np.nan, dtype=float)

            test_labels = participant_labels[test_idx]
            fold_metrics.append(compute_classification_metrics(test_labels, scores))
            oof_y.extend(test_labels.tolist())
            oof_p.extend(scores.tolist())
            tn, fp, fn, tp = confusion_matrix(
                test_labels, (scores >= 0.5).astype(int), labels=[0, 1]
            ).ravel()
            confusion_acc += np.array([tn, fp, fn, tp], dtype=float)

        aggregated = aggregate_fold_metrics(fold_metrics)
        return ModelFamilyResult(
            model_name=model_name,
            base_results={},
            fusion_results={EARLY_FUSION_RESULT_KEY: aggregated},
            roc_curves={
                EARLY_FUSION_RESULT_KEY: compute_roc_curve(
                    np.array(oof_y), np.array(oof_p)
                )
            },
            pr_curves={
                EARLY_FUSION_RESULT_KEY: compute_pr_curve(
                    np.array(oof_y), np.array(oof_p)
                )
            },
            confusion={
                EARLY_FUSION_RESULT_KEY: {
                    "tn": float(confusion_acc[0]),
                    "fp": float(confusion_acc[1]),
                    "fn": float(confusion_acc[2]),
                    "tp": float(confusion_acc[3]),
                }
            },
            best_params={EARLY_FUSION_RESULT_KEY: best_params},
        )

    def _build_ranking(
        self, model_results: List[ModelFamilyResult]
    ) -> List[Dict[str, object]]:
        ranking: List[Dict[str, object]] = []
        for result in model_results:
            metrics = result.fusion_results.get(EARLY_FUSION_RESULT_KEY, {})
            ranking.append(
                {
                    "model_name": result.model_name,
                    "fusion": EARLY_FUSION_RESULT_KEY,
                    "roc_auc_mean": metrics.get("roc_auc_mean", float("nan")),
                    "pr_auc_mean": metrics.get("pr_auc_mean", float("nan")),
                    "balanced_accuracy_mean": metrics.get(
                        "balanced_accuracy_mean", float("nan")
                    ),
                    "score": ranking_score(metrics),
                }
            )
        ranking.sort(key=lambda entry: entry["score"], reverse=True)
        return ranking

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[classification] {message}")
