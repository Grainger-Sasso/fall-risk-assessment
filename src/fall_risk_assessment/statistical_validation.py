"""Unbiased performance estimation and statistical inference for fall-risk models.

Why this exists
---------------
The standard evaluator ranks model families on the same cross-validation folds
that produce the reported metrics. Picking the best of K families on those
folds and then quoting that family's fold metrics incurs selection optimism
("winner's curse"). For claims about whether the pipeline has predictive value
at all, three things are needed:

1. **Fully nested cross-validation** (`nested_family_selection_cv`): model
   family selection (and each family's internal hyperparameter tuning) happens
   inside every outer training fold, so outer test folds measure the *whole
   selection procedure*, not a post-hoc favorite.
2. **Bootstrap confidence intervals** (`bootstrap_metric_ci`): participants are
   the exchangeable unit; resampling them quantifies how much the point
   estimate owes to this particular cohort of ~70 people.
3. **A permutation test** (`permutation_test`): rerunning the identical
   pipeline on label-shuffled data builds the null distribution of the
   performance statistic; the p-value answers "could this AUC arise with no
   real association between gait features and faller status?"

All splits operate at the participant level (one row per participant), so no
leakage is possible by construction.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.classification.data.classification_dataset import (
    EarlyFusionParticipantDataset,
)
from src.classification.evaluation.cross_validation import (
    iter_participant_folds,
    resolve_n_splits,
)
from src.classification.evaluation.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
)
from src.classification.models.model_factory import available_model_names, create_model


@dataclass
class NestedCVResult:
    """Out-of-fold predictions and diagnostics from nested family-selection CV."""

    oof_y: np.ndarray
    oof_p: np.ndarray
    oof_participant_index: np.ndarray
    fold_metrics: List[Dict[str, float]]
    selected_family_counts: Dict[str, int]
    n_splits: int
    n_repeats: int

    @property
    def roc_auc(self) -> float:
        return float(roc_auc_score(self.oof_y, self.oof_p))

    @property
    def pr_auc(self) -> float:
        return float(average_precision_score(self.oof_y, self.oof_p))

    def participant_level_probabilities(
        self, n_participants: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Average each participant's OOF probability across CV repeats.

        Returns (y, p) with one entry per participant that appeared in at
        least one test fold (all of them, for k-fold).
        """
        y_out = np.full(n_participants, -1, dtype=int)
        p_sum = np.zeros(n_participants, dtype=float)
        p_count = np.zeros(n_participants, dtype=float)
        for y_val, p_val, idx in zip(
            self.oof_y, self.oof_p, self.oof_participant_index
        ):
            if not np.isfinite(p_val):
                continue
            y_out[idx] = y_val
            p_sum[idx] += p_val
            p_count[idx] += 1.0
        seen = p_count > 0
        return y_out[seen], p_sum[seen] / p_count[seen]

    def summary(self) -> Dict[str, object]:
        aggregated = aggregate_fold_metrics(self.fold_metrics)
        return {
            "oof_roc_auc": self.roc_auc,
            "oof_pr_auc": self.pr_auc,
            "fold_metrics": aggregated,
            "selected_family_counts": dict(self.selected_family_counts),
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
        }


def _family_inner_score(
    X: np.ndarray,
    y: np.ndarray,
    family: str,
    inner_splits: int,
    random_state: int,
    fast_mode: bool,
) -> float:
    """Mean ROC-AUC of one family across an inner CV on the outer-train data."""
    splits = resolve_n_splits(y, desired=inner_splits)
    scores: List[float] = []
    for train_idx, val_idx in iter_participant_folds(
        y, splits, n_repeats=1, random_state=random_state
    ):
        model = create_model(
            family,
            random_state=random_state,
            fast_mode=fast_mode,
            enable_tuning=True,
        )
        try:
            model.fit(X[train_idx], y[train_idx], groups=None)
            probs = model.predict_proba(X[val_idx])
        except Exception:
            continue
        if len(np.unique(y[val_idx])) < 2:
            continue
        scores.append(float(roc_auc_score(y[val_idx], probs)))
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def nested_family_selection_cv(
    dataset: EarlyFusionParticipantDataset,
    model_names: Optional[List[str]] = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    fast_mode: bool = False,
    verbose: bool = False,
    log: Callable[[str], None] = print,
) -> NestedCVResult:
    """Nested CV where the model family itself is selected inside each fold."""
    families = model_names or available_model_names()
    labels = np.asarray(dataset.y, dtype=int)
    X = dataset.X
    outer_splits = resolve_n_splits(labels, desired=n_splits)

    oof_y: List[int] = []
    oof_p: List[float] = []
    oof_index: List[int] = []
    fold_metrics: List[Dict[str, float]] = []
    family_counts: Dict[str, int] = {family: 0 for family in families}

    for fold_number, (train_idx, test_idx) in enumerate(
        iter_participant_folds(labels, outer_splits, n_repeats, random_state)
    ):
        inner_scores = {
            family: _family_inner_score(
                X[train_idx],
                labels[train_idx],
                family,
                inner_splits=inner_splits,
                random_state=random_state + fold_number,
                fast_mode=fast_mode,
            )
            for family in families
        }
        best_family = max(inner_scores, key=inner_scores.get)
        family_counts[best_family] += 1

        model = create_model(
            best_family,
            random_state=random_state,
            fast_mode=fast_mode,
            enable_tuning=True,
        )
        try:
            model.fit(X[train_idx], labels[train_idx], groups=None)
            probs = model.predict_proba(X[test_idx])
        except Exception:
            probs = np.full(len(test_idx), np.nan, dtype=float)

        fold_metrics.append(compute_classification_metrics(labels[test_idx], probs))
        oof_y.extend(labels[test_idx].tolist())
        oof_p.extend(probs.tolist())
        oof_index.extend(np.asarray(test_idx, dtype=int).tolist())
        if verbose:
            log(
                f"[nested-cv] fold {fold_number + 1}: selected={best_family} "
                f"inner_auc={inner_scores[best_family]:.3f}"
            )

    return NestedCVResult(
        oof_y=np.asarray(oof_y, dtype=int),
        oof_p=np.asarray(oof_p, dtype=float),
        oof_participant_index=np.asarray(oof_index, dtype=int),
        fold_metrics=fold_metrics,
        selected_family_counts=family_counts,
        n_splits=outer_splits,
        n_repeats=n_repeats,
    )


def bootstrap_metric_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI over participants for a probability metric."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(random_state)
    estimates: List[float] = []
    n = len(y)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        y_sample = y[sample_idx]
        if len(np.unique(y_sample)) < 2:
            continue
        estimates.append(float(metric(y_sample, p[sample_idx])))
    estimates_arr = np.asarray(estimates, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point_estimate": float(metric(y, p)),
        "ci_lower": float(np.quantile(estimates_arr, alpha)),
        "ci_upper": float(np.quantile(estimates_arr, 1.0 - alpha)),
        "confidence": confidence,
        "n_bootstrap_effective": float(len(estimates_arr)),
    }


def permutation_test(
    dataset: EarlyFusionParticipantDataset,
    model_names: Optional[List[str]] = None,
    n_permutations: int = 200,
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    fast_mode: bool = True,
    verbose: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Permutation test of the full nested-selection pipeline.

    The observed statistic and every null statistic are computed with the
    *identical* procedure (nested family selection, single repeat), so the
    comparison is exchangeable. ``fast_mode`` shrinks hyperparameter grids for
    tractability; this affects observed and null runs equally.
    """

    def _statistic(labels: np.ndarray, seed: int) -> float:
        permuted = EarlyFusionParticipantDataset(
            X=dataset.X,
            y=labels,
            participant_ids=list(dataset.participant_ids),
            stride_feature_names=list(dataset.stride_feature_names),
            epoch_feature_names=list(dataset.epoch_feature_names),
            aggregation=dataset.aggregation,
        )
        result = nested_family_selection_cv(
            permuted,
            model_names=model_names,
            n_splits=n_splits,
            n_repeats=1,
            inner_splits=inner_splits,
            random_state=seed,
            fast_mode=fast_mode,
        )
        return result.roc_auc

    labels = np.asarray(dataset.y, dtype=int)
    observed = _statistic(labels, random_state)
    if verbose:
        log(f"[permutation] observed AUC (matched procedure): {observed:.3f}")

    rng = np.random.default_rng(random_state)
    null_stats: List[float] = []
    for permutation_number in range(n_permutations):
        shuffled = rng.permutation(labels)
        null_stats.append(_statistic(shuffled, random_state + permutation_number + 1))
        if verbose and (permutation_number + 1) % 10 == 0:
            log(
                f"[permutation] {permutation_number + 1}/{n_permutations} "
                f"null_mean={np.mean(null_stats):.3f}"
            )

    null_arr = np.asarray(null_stats, dtype=float)
    p_value = float((1 + int(np.sum(null_arr >= observed))) / (len(null_arr) + 1))
    return {
        "observed_auc": float(observed),
        "null_auc_mean": float(np.mean(null_arr)),
        "null_auc_std": float(np.std(null_arr, ddof=1)),
        "null_auc_q95": float(np.quantile(null_arr, 0.95)),
        "p_value": p_value,
        "n_permutations": int(len(null_arr)),
        "null_aucs": [float(value) for value in null_arr],
    }
