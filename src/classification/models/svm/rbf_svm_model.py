from typing import Dict, List

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from src.classification.feature_preprocessor.feature_preprocessor import PreparedDataset
from src.classification.models.base_classifier_model import (
    BaseClassifierModel,
    ModelEvaluationResult,
)


class RbfSvmModel(BaseClassifierModel):
    """RBF-kernel SVM with repeated CV evaluation."""

    def __init__(self, random_state: int = 42, fast_mode: bool = False):
        self.random_state = random_state
        self.fast_mode = fast_mode
        self.final_best_params: Dict[str, float] = {}

    @property
    def model_name(self) -> str:
        return "svm_rbf"

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

    def _build_search(self) -> GridSearchCV:
        inner_cv_splits = 3 if self.fast_mode else 5
        c_grid = [0.1, 1.0, 10.0] if self.fast_mode else [0.1, 1.0, 10.0, 100.0]
        gamma_grid = ["scale", 0.1] if self.fast_mode else ["scale", 0.01, 0.1, 1.0]
        base_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        return GridSearchCV(
            estimator=base_pipeline,
            param_grid={
                "clf__C": c_grid,
                "clf__gamma": gamma_grid,
            },
            scoring="roc_auc",
            cv=StratifiedKFold(
                n_splits=inner_cv_splits,
                shuffle=True,
                random_state=self.random_state,
            ),
            n_jobs=-1,
            refit=True,
        )

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
        best_params_history: List[Dict] = []
        for train_index, test_index in splitter.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            search = clone(self._build_search())
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]
            fold_metrics.append(self._compute_fold_metrics(y_test, y_pred, y_proba))
            best_params_history.append(search.best_params_)

        metrics = self._aggregate_metrics(fold_metrics)
        metrics["n_splits"] = float(n_splits)
        metrics["n_repeats"] = float(n_repeats)

        c_values = [params["clf__C"] for params in best_params_history]
        gamma_values = [params["clf__gamma"] for params in best_params_history]
        self.final_best_params = {
            "clf__C_mode": max(set(c_values), key=c_values.count),
            "clf__gamma_mode": max(set(gamma_values), key=gamma_values.count),
        }
        return ModelEvaluationResult(
            model_name=self.model_name,
            metrics=metrics,
            best_hyperparameters=self.final_best_params,
        )
