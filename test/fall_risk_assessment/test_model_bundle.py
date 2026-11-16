"""Tests for FallRiskModelBundle save/load round-trips and prediction guards."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_types.feature.feature_type import FeatureType
from src.fall_risk_assessment.model_bundle import (
    BUNDLE_SCHEMA_VERSION,
    ExtractionConfig,
    FallRiskModelBundle,
    MANIFEST_FILE_NAME,
    RISK_CATEGORY_HIGH,
    RISK_CATEGORY_LOW,
    RISK_CATEGORY_MODERATE,
    RiskThresholds,
)

STRIDE_NAMES = [FeatureType.STRIDE_TIME, FeatureType.STANCE_TIME]
EPOCH_NAMES = [FeatureType.BOUT_DURATION]


def _fitted_model(n_features: int):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n_features))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return model


def _bundle() -> FallRiskModelBundle:
    return FallRiskModelBundle(
        model=_fitted_model(len(STRIDE_NAMES) + len(EPOCH_NAMES)),
        model_name="logistic_regression_elastic_net",
        stride_feature_names=list(STRIDE_NAMES),
        epoch_feature_names=list(EPOCH_NAMES),
        extraction_config=ExtractionConfig(profile="treadmill"),
        thresholds=RiskThresholds(decision_threshold=0.4),
        training_provenance={"n_training_participants": 40},
        cohort_feature_stats={
            "stride time": {"n": 40.0, "q05": 0.9, "q50": 1.1, "q95": 1.4}
        },
    )


class TestRiskThresholds:
    def test_categorize_bands(self):
        thresholds = RiskThresholds(low_boundary=0.33, high_boundary=0.66)
        assert thresholds.categorize(0.1) == RISK_CATEGORY_LOW
        assert thresholds.categorize(0.5) == RISK_CATEGORY_MODERATE
        assert thresholds.categorize(0.9) == RISK_CATEGORY_HIGH

    def test_round_trip(self):
        thresholds = RiskThresholds(0.42, 0.3, 0.7)
        restored = RiskThresholds.from_dict(thresholds.to_dict())
        assert restored.decision_threshold == pytest.approx(0.42)
        assert restored.low_boundary == pytest.approx(0.3)
        assert restored.high_boundary == pytest.approx(0.7)


class TestFallRiskModelBundle:
    def test_save_load_round_trip(self, tmp_path):
        bundle = _bundle()
        bundle_dir = bundle.save(tmp_path / "bundle")

        restored = FallRiskModelBundle.load(bundle_dir)

        assert restored.model_name == bundle.model_name
        assert restored.schema_version == BUNDLE_SCHEMA_VERSION
        assert restored.stride_feature_names == STRIDE_NAMES
        assert restored.epoch_feature_names == EPOCH_NAMES
        assert restored.extraction_config.profile == "treadmill"
        assert restored.thresholds.decision_threshold == pytest.approx(0.4)
        assert restored.cohort_feature_stats["stride time"]["q50"] == pytest.approx(1.1)

        vector = np.array([1.0, 0.5, 2.0])
        assert restored.predict_probability(vector) == pytest.approx(
            bundle.predict_probability(vector)
        )

    def test_predict_probability_rejects_wrong_length(self):
        bundle = _bundle()
        with pytest.raises(ValueError, match="expects"):
            bundle.predict_probability(np.array([1.0, 2.0]))

    def test_load_rejects_unknown_schema_version(self, tmp_path):
        bundle_dir = _bundle().save(tmp_path / "bundle")
        manifest_path = bundle_dir / MANIFEST_FILE_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = "0.0"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="schema version"):
            FallRiskModelBundle.load(bundle_dir)

    def test_load_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FallRiskModelBundle.load(tmp_path / "nope")

    def test_feature_names_order_is_stride_then_epoch(self):
        bundle = _bundle()
        assert bundle.feature_names == STRIDE_NAMES + EPOCH_NAMES
        assert bundle.n_features == 3
