"""Tests for nested CV, bootstrap CIs, and the permutation test.

The key property under test: on pure-noise data the machinery must NOT report
predictive value (nested AUC near 0.5, permutation p-value large), and on
clearly separable data it must (high AUC, small p-value). This is exactly the
guarantee that makes "definitively has / does not have predictive value"
claims possible.
"""

import numpy as np
import pytest

from src.classification.data.classification_dataset import (
    EarlyFusionParticipantDataset,
)
from src.data_types.feature.feature_type import FeatureType
from src.fall_risk_assessment.statistical_validation import (
    bootstrap_metric_ci,
    nested_family_selection_cv,
    permutation_test,
)

FAST_FAMILIES = ["logistic_regression_elastic_net"]


def _dataset(n=40, n_features=4, signal=0.0, seed=0) -> EarlyFusionParticipantDataset:
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2), dtype=int)
    X = rng.normal(size=(n, n_features))
    X[:, 0] += signal * y  # feature 0 carries the label signal when signal > 0
    return EarlyFusionParticipantDataset(
        X=X,
        y=y,
        participant_ids=[f"P{i:03d}" for i in range(n)],
        stride_feature_names=[FeatureType.STRIDE_TIME, FeatureType.STANCE_TIME],
        epoch_feature_names=[FeatureType.BOUT_DURATION, FeatureType.BOUT_STEPS],
        aggregation="mean",
    )


class TestNestedFamilySelectionCV:
    def test_separable_data_scores_high(self):
        result = nested_family_selection_cv(
            _dataset(signal=3.0),
            model_names=FAST_FAMILIES,
            n_splits=3,
            n_repeats=1,
            inner_splits=2,
            fast_mode=True,
        )
        assert result.roc_auc > 0.85

    def test_noise_data_scores_near_chance(self):
        result = nested_family_selection_cv(
            _dataset(signal=0.0),
            model_names=FAST_FAMILIES,
            n_splits=3,
            n_repeats=2,
            inner_splits=2,
            fast_mode=True,
        )
        assert 0.3 < result.roc_auc < 0.7

    def test_every_outer_fold_selects_a_family(self):
        result = nested_family_selection_cv(
            _dataset(signal=1.0),
            model_names=FAST_FAMILIES,
            n_splits=3,
            n_repeats=2,
            inner_splits=2,
            fast_mode=True,
        )
        assert sum(result.selected_family_counts.values()) == 3 * 2

    def test_participant_level_probabilities_average_repeats(self):
        dataset = _dataset(signal=2.0)
        result = nested_family_selection_cv(
            dataset,
            model_names=FAST_FAMILIES,
            n_splits=4,
            n_repeats=3,
            inner_splits=2,
            fast_mode=True,
        )
        y, p = result.participant_level_probabilities(dataset.n_participants)
        assert len(y) == dataset.n_participants  # every participant tested
        assert len(p) == len(y)
        assert np.all((p >= 0.0) & (p <= 1.0))


class TestBootstrapMetricCI:
    def test_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(3)
        y = np.array([0, 1] * 30)
        p = np.clip(0.5 + 0.3 * (y - 0.5) + 0.2 * rng.normal(size=60), 0, 1)
        ci = bootstrap_metric_ci(y, p, n_bootstrap=500)
        assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]
        assert 0.0 <= ci["ci_lower"] < ci["ci_upper"] <= 1.0

    def test_perfect_separation_gives_tight_high_ci(self):
        y = np.array([0] * 20 + [1] * 20)
        p = np.concatenate([np.linspace(0, 0.4, 20), np.linspace(0.6, 1.0, 20)])
        ci = bootstrap_metric_ci(y, p, n_bootstrap=300)
        assert ci["point_estimate"] == pytest.approx(1.0)
        assert ci["ci_lower"] == pytest.approx(1.0)


class TestPermutationTest:
    def test_separable_data_yields_small_p(self):
        result = permutation_test(
            _dataset(n=30, signal=3.0),
            model_names=FAST_FAMILIES,
            n_permutations=19,
            n_splits=3,
            inner_splits=2,
            fast_mode=True,
        )
        # With 19 permutations the smallest attainable p is 1/20.
        assert result["p_value"] == pytest.approx(0.05)
        assert result["observed_auc"] > result["null_auc_mean"]

    def test_noise_data_yields_large_p(self):
        result = permutation_test(
            _dataset(n=30, signal=0.0, seed=11),
            model_names=FAST_FAMILIES,
            n_permutations=19,
            n_splits=3,
            inner_splits=2,
            fast_mode=True,
        )
        assert result["p_value"] > 0.1

    def test_null_distribution_is_recorded(self):
        result = permutation_test(
            _dataset(n=30, signal=1.0),
            model_names=FAST_FAMILIES,
            n_permutations=5,
            n_splits=3,
            inner_splits=2,
            fast_mode=True,
        )
        assert len(result["null_aucs"]) == 5
        assert result["n_permutations"] == 5
