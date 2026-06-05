from typing import Dict

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.classification.models.base_classifier_model import BaseClassifierModel


class LightGbmShallowModel(BaseClassifierModel):
    """Shallow LightGBM base classifier."""

    @property
    def model_name(self) -> str:
        return "lightgbm_shallow"

    def build_pipeline(self) -> Pipeline:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "lightgbm is required for LightGbmShallowModel. "
                "Install with `pip install lightgbm`."
            ) from exc

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "clf",
                    LGBMClassifier(
                        objective="binary",
                        class_weight="balanced",
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    def param_grid(self) -> Dict[str, list]:
        if self.fast_mode:
            return {
                "clf__n_estimators": [50, 100],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2],
                "clf__num_leaves": [7],
                "clf__min_child_samples": [10],
                "clf__reg_lambda": [0.0, 1.0],
            }
        return {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [2, 3],
            "clf__num_leaves": [7, 15],
            "clf__min_child_samples": [5, 10, 20],
            "clf__reg_lambda": [0.0, 1.0, 5.0],
        }
