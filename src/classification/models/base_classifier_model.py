from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline


class BaseClassifierModel(ABC):
    """
    Adapter wrapping a single sklearn pipeline family with a uniform
    ``fit`` / ``predict_proba`` contract for sample-level base classification.

    Hyperparameter tuning (when enabled) runs a grouped inner cross-validation
    over the training samples so participants are not split across tuning folds.
    """

    def __init__(
        self,
        random_state: int = 42,
        fast_mode: bool = False,
        enable_tuning: bool = True,
    ) -> None:
        self.random_state = random_state
        self.fast_mode = fast_mode
        self.enable_tuning = enable_tuning
        self.best_params_: Dict[str, object] = {}
        self._fitted_estimator: Optional[Pipeline] = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable identifier used in reports and the factory."""

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Return a fresh imputation/scaling/classifier pipeline."""

    @abstractmethod
    def param_grid(self) -> Dict[str, list]:
        """Hyperparameter grid for tuning (empty disables the search)."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> "BaseClassifierModel":
        pipeline = self.build_pipeline()
        grid = self.param_grid() if self.enable_tuning else {}

        if grid:
            inner_cv, fit_groups = self._inner_cv(y, groups)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y, groups=fit_groups)
            self._fitted_estimator = search.best_estimator_
            self.best_params_ = dict(search.best_params_)
        else:
            estimator = clone(pipeline)
            estimator.fit(X, y)
            self._fitted_estimator = estimator
            self.best_params_ = {}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._fitted_estimator is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")
        return self._fitted_estimator.predict_proba(X)[:, 1]

    def _inner_cv(self, y: np.ndarray, groups: Optional[np.ndarray]):
        desired = 3 if self.fast_mode else 5
        y = np.asarray(y).astype(int)
        classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = int(class_counts.min()) if len(classes) else 0

        if groups is not None:
            groups = np.asarray(groups, dtype=object)
            groups_per_class = [
                len(np.unique(groups[y == cls])) for cls in classes
            ]
            min_groups = min(groups_per_class) if groups_per_class else 0
            n_splits = max(2, min(desired, min_groups))
            if min_groups >= 2:
                return (
                    StratifiedGroupKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=self.random_state,
                    ),
                    groups,
                )

        n_splits = max(2, min(desired, min_class_count))
        return (
            StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            ),
            None,
        )
