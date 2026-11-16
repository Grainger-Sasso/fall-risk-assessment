"""Tests for FallRiskModelTrainer helpers (no database required)."""

import numpy as np
import pytest

from src.classification.data.classification_dataset import (
    EarlyFusionParticipantDataset,
)
from src.data_types.feature.feature_type import FeatureType
from src.fall_risk_assessment.training import FallRiskModelTrainer


def _dataset(participant_ids, labels, n_features=4) -> EarlyFusionParticipantDataset:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(len(participant_ids), n_features))
    return EarlyFusionParticipantDataset(
        X=X,
        y=np.asarray(labels, dtype=int),
        participant_ids=list(participant_ids),
        stride_feature_names=[FeatureType.STRIDE_TIME, FeatureType.STANCE_TIME],
        epoch_feature_names=[FeatureType.BOUT_DURATION, FeatureType.BOUT_STEPS],
        aggregation="mean",
    )


def _trainer(**overrides) -> FallRiskModelTrainer:
    defaults = dict(
        sqlite_db_path="unused.db",
        output_dir="unused",
        fast_mode=True,
    )
    defaults.update(overrides)
    return FallRiskModelTrainer(**defaults)


class TestExcludeHoldouts:
    def test_removes_only_named_participants(self):
        dataset = _dataset(["P1", "P2", "P3", "P4"], [0, 1, 0, 1])
        trainer = _trainer(holdout_participant_ids=["P2", "P4"])

        filtered, excluded = trainer._exclude_holdouts(dataset)

        assert excluded == ["P2", "P4"]
        assert filtered.participant_ids == ["P1", "P3"]
        assert filtered.y.tolist() == [0, 0]
        assert filtered.X.shape == (2, 4)

    def test_no_holdouts_returns_same_dataset(self):
        dataset = _dataset(["P1", "P2"], [0, 1])
        trainer = _trainer()
        filtered, excluded = trainer._exclude_holdouts(dataset)
        assert excluded == []
        assert filtered is dataset

    def test_unknown_holdout_is_reported_not_fatal(self, capsys):
        dataset = _dataset(["P1", "P2"], [0, 1])
        trainer = _trainer(holdout_participant_ids=["P9"])
        filtered, excluded = trainer._exclude_holdouts(dataset)
        assert excluded == []
        assert filtered.participant_ids == ["P1", "P2"]
        assert "P9" in capsys.readouterr().out


class TestDeriveThresholds:
    def test_picks_separating_threshold(self):
        oof_y = np.array([0, 0, 0, 1, 1, 1])
        oof_p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thresholds = FallRiskModelTrainer._derive_thresholds(oof_y, oof_p)
        assert 0.3 < thresholds.decision_threshold <= 0.7

    def test_degenerate_labels_fall_back_to_default(self):
        thresholds = FallRiskModelTrainer._derive_thresholds(
            np.array([1, 1]), np.array([0.6, 0.7])
        )
        assert thresholds.decision_threshold == pytest.approx(0.5)

    def test_ignores_non_finite_probabilities(self):
        oof_y = np.array([0, 0, 1, 1, 0, 1])
        oof_p = np.array([0.1, 0.2, 0.9, 0.8, np.nan, np.nan])
        thresholds = FallRiskModelTrainer._derive_thresholds(oof_y, oof_p)
        assert 0.2 < thresholds.decision_threshold <= 0.8


class TestCohortFeatureStats:
    def test_stats_cover_all_features_in_order(self):
        dataset = _dataset(["P1", "P2", "P3"], [0, 1, 0])
        trainer = _trainer()
        stats = trainer._cohort_feature_stats(dataset)
        assert list(stats.keys()) == [
            FeatureType.STRIDE_TIME.value,
            FeatureType.STANCE_TIME.value,
            FeatureType.BOUT_DURATION.value,
            FeatureType.BOUT_STEPS.value,
        ]
        entry = stats[FeatureType.STRIDE_TIME.value]
        assert entry["n"] == 3.0
        assert entry["q05"] <= entry["q50"] <= entry["q95"]

    def test_all_nan_feature_records_zero_count(self):
        dataset = _dataset(["P1", "P2"], [0, 1])
        dataset.X[:, 0] = np.nan
        trainer = _trainer()
        stats = trainer._cohort_feature_stats(dataset)
        assert stats[FeatureType.STRIDE_TIME.value] == {"n": 0.0}


class TestFitCalibrated:
    def test_produces_probabilities_in_unit_interval(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(24, 4))
        y = (X[:, 0] + 0.2 * rng.normal(size=24) > 0).astype(int)
        trainer = _trainer(n_splits=3, n_repeats=1)

        model = trainer._fit_calibrated(X, y, "logistic_regression_elastic_net")

        probs = model.predict_proba(X)[:, 1]
        assert probs.shape == (24,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
