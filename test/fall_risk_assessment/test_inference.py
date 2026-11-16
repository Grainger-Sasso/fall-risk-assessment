"""Tests for FallRiskAssessor feature vectorization and assessment output."""

import numpy as np
import pytest

from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.fall_risk_assessment.inference import FallRiskAssessor
from src.fall_risk_assessment.model_bundle import (
    ExtractionConfig,
    FallRiskModelBundle,
    RiskThresholds,
)
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier

STRIDE_NAMES = [FeatureType.STRIDE_TIME, FeatureType.STANCE_TIME]
EPOCH_NAMES = [FeatureType.BOUT_DURATION]


class _ConstantModel:
    """Stub estimator returning a fixed faller probability."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, X):
        rows = np.asarray(X).shape[0]
        return np.column_stack(
            [np.full(rows, 1.0 - self.probability), np.full(rows, self.probability)]
        )


def _bout_features(basis, feature_names, values) -> BoutFeatures:
    """Build a (1 bout, M features, N samples) tensor from a values matrix."""
    tensor = np.asarray(values, dtype=float)[np.newaxis, :, :]
    num_samples = tensor.shape[2]
    return BoutFeatures(
        sample_basis=basis,
        features=tensor,
        bout_starts=np.array([0.0]),
        bout_ends=np.array([60.0]),
        feature_names=list(feature_names),
        sample_starts=np.arange(num_samples, dtype=float),
        sample_ends=np.arange(num_samples, dtype=float) + 1.0,
        units=["s"] * len(feature_names),
    )


def _record_features(stride_values, epoch_values, stride_names=None) -> RecordFeatures:
    return RecordFeatures(
        stride_features=_bout_features(
            SampleBasis.STRIDE, stride_names or STRIDE_NAMES, stride_values
        ),
        epoch_features=_bout_features(SampleBasis.EPOCH, EPOCH_NAMES, epoch_values),
        feature_metadata=FeatureMetadata(
            feature_identifier=FeatureIdentifier("feature_test"),
            user_identifier=UserIdentifier("P1"),
            imu_data_identifier=IMUDataIdentifier("imu_test"),
        ),
    )


def _bundle(probability=0.8) -> FallRiskModelBundle:
    return FallRiskModelBundle(
        model=_ConstantModel(probability),
        model_name="stub",
        stride_feature_names=list(STRIDE_NAMES),
        epoch_feature_names=list(EPOCH_NAMES),
        extraction_config=ExtractionConfig(profile="treadmill"),
        thresholds=RiskThresholds(decision_threshold=0.5),
        cohort_feature_stats={
            FeatureType.STRIDE_TIME.value: {
                "n": 30.0,
                "q05": 0.9,
                "q25": 1.0,
                "q50": 1.1,
                "q75": 1.2,
                "q95": 1.4,
            }
        },
    )


def _assessor(bundle=None) -> FallRiskAssessor:
    """Build an assessor without triggering extraction-pipeline construction."""
    assessor = FallRiskAssessor.__new__(FallRiskAssessor)
    assessor.bundle = bundle or _bundle()
    assessor.pipeline = None
    return assessor


class TestAssessFeatures:
    def test_high_probability_classifies_high_risk_faller(self):
        assessor = _assessor(_bundle(probability=0.8))
        record = _record_features(
            stride_values=[[1.0, 1.2], [0.7, 0.8]],
            epoch_values=[[30.0, 40.0]],
        )
        assessment = assessor.assess_features(record)

        assert assessment.user_id == "P1"
        assert assessment.imu_record_id == "imu_test"
        assert assessment.risk_probability == pytest.approx(0.8)
        assert assessment.risk_category == "high"
        assert assessment.predicted_faller is True

    def test_feature_values_are_nanmean_over_samples(self):
        assessor = _assessor(_bundle(probability=0.2))
        record = _record_features(
            stride_values=[[1.0, 2.0], [0.5, np.nan]],
            epoch_values=[[10.0, 20.0]],
        )
        assessment = assessor.assess_features(record)

        values = assessment.feature_values
        assert values[FeatureType.STRIDE_TIME.value] == pytest.approx(1.5)
        assert values[FeatureType.STANCE_TIME.value] == pytest.approx(0.5)
        assert values[FeatureType.BOUT_DURATION.value] == pytest.approx(15.0)
        assert assessment.risk_category == "low"
        assert assessment.predicted_faller is False

    def test_missing_expected_feature_is_reported_and_nan(self):
        assessor = _assessor(_bundle(probability=0.5))
        # Record only produced STRIDE_TIME; STANCE_TIME is absent.
        record = _record_features(
            stride_values=[[1.0, 1.1]],
            epoch_values=[[30.0]],
            stride_names=[FeatureType.STRIDE_TIME],
        )
        assessment = assessor.assess_features(record)

        assert assessment.feature_values[FeatureType.STANCE_TIME.value] is None
        assert (
            FeatureType.STANCE_TIME.value
            in assessment.stride_quality.missing_features
        )
        assert assessment.stride_quality.n_features_observed == 1

    def test_quality_counts_usable_samples(self):
        assessor = _assessor()
        record = _record_features(
            stride_values=[[1.0, np.nan, 1.2], [0.7, np.nan, 0.9]],
            epoch_values=[[30.0]],
        )
        assessment = assessor.assess_features(record)
        # middle sample slot is all-NaN padding -> dropped
        assert assessment.stride_quality.n_usable_samples == 2
        assert assessment.epoch_quality.n_usable_samples == 1


class TestFeaturePercentiles:
    def test_percentile_interpolates_within_cohort(self):
        assessor = _assessor()
        percentiles = assessor._feature_percentiles(
            {FeatureType.STRIDE_TIME.value: 1.1}
        )
        assert percentiles[FeatureType.STRIDE_TIME.value] == pytest.approx(50.0)

    def test_value_above_cohort_clamps_to_95(self):
        assessor = _assessor()
        percentiles = assessor._feature_percentiles(
            {FeatureType.STRIDE_TIME.value: 99.0}
        )
        assert percentiles[FeatureType.STRIDE_TIME.value] == pytest.approx(95.0)

    def test_unknown_feature_or_none_value_yields_none(self):
        assessor = _assessor()
        percentiles = assessor._feature_percentiles(
            {"unknown feature": 1.0, FeatureType.STRIDE_TIME.value: None}
        )
        assert percentiles["unknown feature"] is None
        assert percentiles[FeatureType.STRIDE_TIME.value] is None
