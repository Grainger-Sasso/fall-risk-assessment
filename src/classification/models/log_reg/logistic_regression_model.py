from typing import Dict, List

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.classification.feature_preprocessor.feature_preprocessor import PreparedDataset
from src.classification.models.base_classifier_model import (
    BaseClassifierModel,
    ModelEvaluationResult,
)


class LogisticRegressionElasticNetModel(BaseClassifierModel):
    """Elastic-net logistic regression with repeated CV evaluation."""

    def __init__(self, random_state: int = 42, fast_mode: bool = False):
        self.random_state = random_state
        self.fast_mode = fast_mode
        self.final_best_params: Dict[str, float] = {}
        l1_ratios = [0.5] if fast_mode else [0.1, 0.5, 0.7, 0.9]
        inner_cv_splits = 3 if fast_mode else 5
        max_iter = 3000 if fast_mode else 10000
        self.pipeline_template = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "clf",
                    LogisticRegressionCV(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratios=l1_ratios,
                        cv=StratifiedKFold(
                            n_splits=inner_cv_splits,
                            shuffle=True,
                            random_state=random_state,
                        ),
                        max_iter=max_iter,
                        scoring="roc_auc",
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                        refit=True,
                    ),
                ),
            ]
        )

    @property
    def model_name(self) -> str:
        return "logistic_regression_elastic_net"

    @staticmethod
    def _compute_fold_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
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
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
        return metrics

    @staticmethod
    def _aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        metric_names = sorted(fold_metrics[0].keys())
        summary: Dict[str, float] = {"n_folds": float(len(fold_metrics))}
        for metric_name in metric_names:
            values = np.array([row[metric_name] for row in fold_metrics], dtype=float)
            summary[f"{metric_name}_mean"] = float(np.nanmean(values))
            summary[f"{metric_name}_std"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            )
        return summary

    def evaluate(
        self,
        prepared_data: PreparedDataset,
        n_splits: int,
        n_repeats: int,
        random_state: int,
    ) -> ModelEvaluationResult:
        X = prepared_data.features
        y = prepared_data.labels
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        fold_metrics: List[Dict[str, float]] = []
        best_cs: List[float] = []
        best_l1_ratios: List[float] = []
        for train_index, test_index in splitter.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_pipeline = clone(self.pipeline_template)
            fold_pipeline.fit(X_train, y_train)
            y_pred = fold_pipeline.predict(X_test)
            y_proba = fold_pipeline.predict_proba(X_test)[:, 1]
            fold_metrics.append(self._compute_fold_metrics(y_test, y_pred, y_proba))

            clf: LogisticRegressionCV = fold_pipeline.named_steps["clf"]
            best_cs.append(float(np.ravel(clf.C_)[0]))
            best_l1_ratios.append(float(np.ravel(clf.l1_ratio_)[0]))

        metrics = self._aggregate_metrics(fold_metrics)
        metrics["n_splits"] = float(n_splits)
        metrics["n_repeats"] = float(n_repeats)

        self.final_best_params = {
            "best_C_median": float(np.median(np.array(best_cs))),
            "best_l1_ratio_median": float(np.median(np.array(best_l1_ratios))),
        }
        return ModelEvaluationResult(
            model_name=self.model_name,
            metrics=metrics,
            best_hyperparameters=self.final_best_params,
        )
