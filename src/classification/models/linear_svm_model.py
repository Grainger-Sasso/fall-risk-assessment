from typing import Dict

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.classification.models.base_classifier_model import BaseClassifierModel
from src.classification.models.preprocessing import (
    SVM_MAX_ITER,
    DegenerateSafeRobustScaler,
)


class LinearSvmModel(BaseClassifierModel):
    """Linear-kernel SVM base classifier."""

    @property
    def model_name(self) -> str:
        return "svm_linear"

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", DegenerateSafeRobustScaler()),
                (
                    "clf",
                    SVC(
                        kernel="linear",
                        probability=True,
                        max_iter=SVM_MAX_ITER,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def param_grid(self) -> Dict[str, list]:
        if self.fast_mode:
            return {"clf__C": [0.1, 1.0, 10.0]}
        return {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
