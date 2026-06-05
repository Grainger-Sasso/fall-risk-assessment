"""Participant-level classification metrics (single source of truth)."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

THRESHOLD_METRICS = [
    "balanced_accuracy",
    "precision",
    "f1",
    "sensitivity",
    "specificity",
    "brier_score",
]
RANKING_METRICS = ["roc_auc", "pr_auc"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute threshold and ranking metrics for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tp"] = float(tp)
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a list of per-fold metric dicts into mean/std summaries."""
    if not fold_metrics:
        return {"n_folds": 0.0}
    metric_names = sorted(fold_metrics[0].keys())
    summary: Dict[str, float] = {"n_folds": float(len(fold_metrics))}
    for metric_name in metric_names:
        values = np.array(
            [row.get(metric_name, np.nan) for row in fold_metrics], dtype=float
        )
        summary[f"{metric_name}_mean"] = float(np.nanmean(values))
        summary[f"{metric_name}_std"] = (
            float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        )
    return summary


def compute_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, List[float]]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}


def compute_pr_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, List[float]]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"recall": [], "precision": []}
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {
        "recall": [float(v) for v in recall],
        "precision": [float(v) for v in precision],
    }


def ranking_score(metrics: Dict[str, float]) -> float:
    """Combine ROC-AUC and PR-AUC means into a single ranking score."""
    roc = metrics.get("roc_auc_mean", float("nan"))
    pr = metrics.get("pr_auc_mean", float("nan"))
    if not np.isfinite(roc) and not np.isfinite(pr):
        return -1.0
    roc = roc if np.isfinite(roc) else 0.0
    pr = pr if np.isfinite(pr) else 0.0
    return 0.5 * roc + 0.5 * pr
