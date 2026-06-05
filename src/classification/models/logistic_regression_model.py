from typing import Dict

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.classification.models.base_classifier_model import BaseClassifierModel


class LogisticRegressionElasticNetModel(BaseClassifierModel):
    """Elastic-net logistic regression base classifier."""

    @property
    def model_name(self) -> str:
        return "logistic_regression_elastic_net"

    def build_pipeline(self) -> Pipeline:
        max_iter = 3000 if self.fast_mode else 10000
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        C=1.0,
                        class_weight="balanced",
                        max_iter=max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def param_grid(self) -> Dict[str, list]:
        if self.fast_mode:
            return {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__l1_ratio": [0.5],
            }
        return {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.1, 0.5, 0.9],
        }
