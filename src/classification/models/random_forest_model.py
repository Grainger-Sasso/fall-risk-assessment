from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.classification.models.base_classifier_model import BaseClassifierModel


class RandomForestModel(BaseClassifierModel):
    """Random forest base classifier (robust tree baseline)."""

    @property
    def model_name(self) -> str:
        return "random_forest"

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def param_grid(self) -> Dict[str, list]:
        if self.fast_mode:
            return {
                "clf__n_estimators": [200],
                "clf__max_depth": [None, 5],
                "clf__min_samples_leaf": [1, 5],
            }
        return {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2"],
        }
