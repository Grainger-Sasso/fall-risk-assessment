import unittest

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from test.gait_features.validation.reference_validators import (
    assert_gait_speed_matches_reference,
    median_stride_gait_speed_mps,
)


def _record_with_stride_speeds(speeds: list[float]) -> RecordFeatures:
    sample_count = len(speeds)
    gait_speeds = np.array(speeds, dtype=float).reshape(1, 1, sample_count)
    return RecordFeatures(
        epoch_features=BoutFeatures(
            sample_basis=SampleBasis.EPOCH,
            features=np.array([[[1.0]]]),
            bout_starts=np.array([0.0]),
            bout_ends=np.array([1.0]),
            feature_names=[FeatureType.GAIT_SPEED],
            sample_starts=np.array([0.0]),
            sample_ends=np.array([1.0]),
            units=["m/s"],
        ),
        stride_features=BoutFeatures(
            sample_basis=SampleBasis.STRIDE,
            features=gait_speeds,
            bout_starts=np.array([0.0]),
            bout_ends=np.array([float(sample_count)]),
            feature_names=[FeatureType.GAIT_SPEED],
            sample_starts=np.arange(sample_count, dtype=float),
            sample_ends=np.arange(1, sample_count + 1, dtype=float),
            units=["m/s"],
        ),
        feature_metadata=FeatureMetadata(
            feature_identifier=type("Id", (), {"value": "feature_test"})(),
            user_identifier=type("Id", (), {"value": "CO001"})(),
            imu_data_identifier=type("Id", (), {"value": "imu_test"})(),
        ),
    )


class TestReferenceValidators(unittest.TestCase):
    def test_median_stride_gait_speed_ignores_nan_padding(self):
        record = _record_with_stride_speeds([1.0, 1.2, np.nan, 1.4])
        self.assertAlmostEqual(median_stride_gait_speed_mps(record), 1.2)

    def test_gait_speed_matches_reference_within_tolerance(self):
        record = _record_with_stride_speeds([1.22, 1.24, 1.26])
        assert_gait_speed_matches_reference(
            record,
            "CO001",
            reference_speed_mps=1.24,
            tolerance_relative=0.05,
        )

    def test_gait_speed_reference_failure_reports_values(self):
        record = _record_with_stride_speeds([2.0, 2.1, 2.2])
        with self.assertRaisesRegex(AssertionError, "GAIT_SPEED mismatch for CO001"):
            assert_gait_speed_matches_reference(
                record,
                "CO001",
                reference_speed_mps=1.24,
                tolerance_relative=0.25,
            )


if __name__ == "__main__":
    unittest.main()
