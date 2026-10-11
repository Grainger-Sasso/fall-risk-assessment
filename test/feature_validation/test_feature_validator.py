import unittest

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.feature_validation.feature_validator import validate_record
from src.feature_validation.range_catalog import default_range_catalog

STRIDE_TYPES = [FeatureType.GAIT_SPEED, FeatureType.CADENCE]


class _FakeIdentifier:
    def __init__(self, value):
        self.value = value


def _record(stride_values: np.ndarray, feature_id="feature_1", user_id="user_1"):
    """Build a record whose stride tensor holds ``stride_values`` shaped (bout, feature, sample)."""
    sample_count = stride_values.shape[2]
    return RecordFeatures(
        epoch_features=BoutFeatures(
            sample_basis=SampleBasis.EPOCH,
            features=np.empty((0, 0, 0)),
            bout_starts=np.array([]),
            bout_ends=np.array([]),
            feature_names=[],
            sample_starts=np.array([]),
            sample_ends=np.array([]),
            units=[],
        ),
        stride_features=BoutFeatures(
            sample_basis=SampleBasis.STRIDE,
            features=stride_values,
            bout_starts=np.zeros(stride_values.shape[0]),
            bout_ends=np.ones(stride_values.shape[0]),
            feature_names=list(STRIDE_TYPES),
            sample_starts=np.arange(sample_count, dtype=float),
            sample_ends=np.arange(1, sample_count + 1, dtype=float),
            units=["m/s", "steps/min"],
        ),
        feature_metadata=FeatureMetadata(
            feature_identifier=_FakeIdentifier(feature_id),
            user_identifier=_FakeIdentifier(user_id),
            imu_data_identifier=_FakeIdentifier("imu_1"),
        ),
    )


class TestValidateRecord(unittest.TestCase):
    def setUp(self):
        self.catalog = default_range_catalog()

    def _validate(self, stride_values):
        return validate_record(
            record=_record(stride_values),
            basis=SampleBasis.STRIDE,
            catalog=self.catalog,
            class_label="faller",
        )

    def test_all_values_in_range(self):
        values = np.array([[[1.2, 1.1], [110.0, 105.0]]])
        result = self._validate(values)
        self.assertEqual(2, result.usable_samples)
        self.assertEqual(0, result.implausible_total)
        self.assertEqual(0, result.atypical_total)
        self.assertEqual(0, result.missing_total)
        self.assertEqual(4, result.finite_total)

    def test_implausible_and_atypical_are_counted_separately(self):
        # 0.0 m/s is impossible; 0.3 m/s is slow but clinically real.
        values = np.array([[[0.0, 0.3, 1.2], [110.0, 110.0, 110.0]]])
        result = self._validate(values)
        speed = next(c for c in result.counts if c.feature_type == FeatureType.GAIT_SPEED)
        self.assertEqual(1, speed.implausible_count)
        self.assertEqual(1, speed.atypical_count)
        self.assertEqual(1, speed.in_range_count)

    def test_padded_slots_are_not_counted_as_missing(self):
        """A fully NaN sample column is structural padding, not missing data."""
        values = np.array([[[1.2, np.nan], [110.0, np.nan]]])
        result = self._validate(values)
        self.assertEqual(1, result.usable_samples)
        self.assertEqual(0, result.missing_total)

    def test_partial_nan_on_a_usable_row_is_missing(self):
        """A usable row where only some features are NaN does count as missing."""
        values = np.array([[[1.2, 1.1], [110.0, np.nan]]])
        result = self._validate(values)
        cadence = next(c for c in result.counts if c.feature_type == FeatureType.CADENCE)
        self.assertEqual(2, cadence.usable_samples)
        self.assertEqual(1, cadence.finite_count)
        self.assertEqual(1, cadence.missing_count)
        self.assertEqual(0.5, cadence.finite_fraction)

    def test_nan_is_never_counted_as_a_range_violation(self):
        values = np.array([[[1.2, np.nan], [110.0, 110.0]]])
        result = self._validate(values)
        speed = next(c for c in result.counts if c.feature_type == FeatureType.GAIT_SPEED)
        self.assertEqual(1, speed.missing_count)
        self.assertEqual(0, speed.implausible_count)
        self.assertEqual(0, speed.atypical_count)

    def test_worst_implausible_tracks_the_furthest_offender(self):
        values = np.array([[[0.05, 0.0, 1.2], [110.0, 110.0, 110.0]]])
        result = self._validate(values)
        speed = next(c for c in result.counts if c.feature_type == FeatureType.GAIT_SPEED)
        self.assertEqual(2, speed.implausible_count)
        self.assertEqual(0.0, speed.worst_implausible)

    def test_multiple_bouts_are_flattened(self):
        values = np.array(
            [
                [[1.2, 1.1], [110.0, 105.0]],
                [[1.3, np.nan], [112.0, np.nan]],
            ]
        )
        result = self._validate(values)
        self.assertEqual(3, result.usable_samples)
        self.assertEqual(6, result.finite_total)

    def test_empty_tensor_yields_empty_result(self):
        result = validate_record(
            record=_record(np.empty((0, 0, 0))),
            basis=SampleBasis.STRIDE,
            catalog=self.catalog,
        )
        self.assertTrue(result.is_empty)
        self.assertEqual(0, result.usable_samples)

    def test_meaningful_fraction_is_one_when_no_range_defined(self):
        """Absent range coverage must never read as a validation failure."""
        record = _record(np.array([[[1.2], [110.0]]]))
        record.stride_features.feature_names = [FeatureType.BOUT_N, FeatureType.CADENCE]
        result = validate_record(
            record=record, basis=SampleBasis.STRIDE, catalog=self.catalog
        )
        uncovered = next(c for c in result.counts if c.feature_type == FeatureType.BOUT_N)
        self.assertFalse(uncovered.has_range)
        self.assertEqual(1.0, uncovered.meaningful_fraction)
        self.assertIn(FeatureType.BOUT_N, result.uncovered_feature_types)

    def test_meaningful_fraction_is_zero_when_nothing_is_finite(self):
        values = np.array([[[np.nan, 1.2], [np.nan, 110.0]]])
        record = _record(values)
        result = validate_record(
            record=record, basis=SampleBasis.STRIDE, catalog=self.catalog
        )
        # Both features have one finite value out of one usable row.
        speed = next(c for c in result.counts if c.feature_type == FeatureType.GAIT_SPEED)
        self.assertEqual(1, speed.finite_count)

    def test_class_label_and_participant_are_carried_through(self):
        result = self._validate(np.array([[[1.2], [110.0]]]))
        self.assertEqual("faller", result.class_label)
        self.assertEqual("user_1", result.participant_id)
        self.assertEqual("feature_1", result.feature_id)


if __name__ == "__main__":
    unittest.main()
