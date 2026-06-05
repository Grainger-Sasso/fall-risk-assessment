from typing import Dict

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from src.classification.models.base_classifier_model import BaseClassifierModel


class RbfSvmModel(BaseClassifierModel):
    """RBF-kernel SVM base classifier."""

    @property
    def model_name(self) -> str:
        return "svm_rbf"

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
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

    def param_grid(self) -> Dict[str, list]:
        if self.fast_mode:
            return {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", 0.1],
            }
        return {
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", 0.01, 0.1, 1.0],
        }
