from typing import Callable, Dict, List

from src.classification.models.base_classifier_model import BaseClassifierModel
from src.classification.models.lightgbm_model import LightGbmShallowModel
from src.classification.models.linear_svm_model import LinearSvmModel
from src.classification.models.logistic_regression_model import (
    LogisticRegressionElasticNetModel,
)
from src.classification.models.random_forest_model import RandomForestModel
from src.classification.models.rbf_svm_model import RbfSvmModel

_MODEL_BUILDERS: Dict[str, Callable[..., BaseClassifierModel]] = {
    "logistic_regression_elastic_net": LogisticRegressionElasticNetModel,
    "svm_linear": LinearSvmModel,
    "svm_rbf": RbfSvmModel,
    "lightgbm_shallow": LightGbmShallowModel,
    "random_forest": RandomForestModel,
}


def available_model_names() -> List[str]:
    return list(_MODEL_BUILDERS.keys())


def create_model(
    model_name: str,
    random_state: int = 42,
    fast_mode: bool = False,
    enable_tuning: bool = True,
) -> BaseClassifierModel:
    if model_name not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available_model_names()}"
        )
    return _MODEL_BUILDERS[model_name](
        random_state=random_state,
        fast_mode=fast_mode,
        enable_tuning=enable_tuning,
    )
