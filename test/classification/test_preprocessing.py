"""Tests for the degenerate-safe scaler and the SVM iteration cap.

Regression guard for the LTMM mean+std+IQR hang: a quantized feature whose
values nearly all coincide had a cross-participant IQR of ~1e-10, scaled one
outlier to ~1e9, and left an uncapped linear SVM spinning indefinitely.
"""

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

from src.classification.models.model_factory import create_model
from src.classification.models.preprocessing import (
    SVM_MAX_ITER,
    DegenerateSafeRobustScaler,
)


def _point_mass_column(n=53, seed=0):
    """0.02 for nearly everyone (with float noise), a few larger outliers."""
    rng = np.random.default_rng(seed)
    col = np.full(n, 0.02) + rng.normal(scale=1e-11, size=n)
    col[:5] = [0.04, 0.04, 0.0475, 0.05, 0.13]
    return col


def _matrix(seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(53, 4))
    return np.column_stack([normal, _point_mass_column(seed=seed)])


def test_well_behaved_columns_match_robust_scaler_exactly():
    X = np.random.default_rng(1).normal(size=(60, 5))
    np.testing.assert_array_equal(
        DegenerateSafeRobustScaler().fit_transform(X),
        RobustScaler().fit_transform(X),
    )


def test_degenerate_column_falls_back_to_std_scaling():
    X = _matrix()
    plain = RobustScaler().fit_transform(X)
    safe_scaler = DegenerateSafeRobustScaler().fit(X)
    safe = safe_scaler.transform(X)

    assert np.abs(plain[:, 4]).max() > 1e6  # the original failure mode
    assert safe_scaler.degenerate_columns_.tolist() == [False] * 4 + [True]
    assert np.abs(safe[:, 4]).max() < 20
    np.testing.assert_allclose(safe[:, :4], plain[:, :4])


def test_constant_column_is_not_flagged():
    X = np.column_stack([np.random.default_rng(2).normal(size=40), np.full(40, 3.0)])
    scaler = DegenerateSafeRobustScaler().fit(X)
    assert not scaler.degenerate_columns_.any()
    assert np.isfinite(scaler.transform(X)).all()


def test_without_scaling_is_untouched():
    X = _matrix()
    scaler = DegenerateSafeRobustScaler(with_scaling=False).fit(X)
    assert not scaler.degenerate_columns_.any()


@pytest.mark.parametrize("family", ["svm_linear", "svm_rbf"])
def test_svm_pipelines_are_capped_and_use_safe_scaler(family):
    pipeline = create_model(family, random_state=0).build_pipeline()
    assert isinstance(pipeline.named_steps["scaler"], DegenerateSafeRobustScaler)
    assert pipeline.named_steps["clf"].max_iter == SVM_MAX_ITER


@pytest.mark.parametrize(
    "family",
    [
        "logistic_regression_elastic_net",
        "svm_linear",
        "svm_rbf",
        "lightgbm_shallow",
        "random_forest",
    ],
)
def test_every_family_uses_safe_scaler(family):
    pipeline = create_model(family, random_state=0).build_pipeline()
    assert isinstance(pipeline.named_steps["scaler"], DegenerateSafeRobustScaler)


def test_linear_svm_converges_on_point_mass_outlier_feature():
    X = _matrix()
    y = np.array([0, 1] * 26 + [1])
    pipeline = create_model("svm_linear", random_state=0).build_pipeline()
    pipeline.set_params(clf__C=100.0, clf__probability=False)
    pipeline.fit(X, y)
    assert int(np.max(pipeline.named_steps["clf"].n_iter_)) < SVM_MAX_ITER
