import unittest

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.metadata.feature_metadata import FeatureMetadata
from src.data_model.features.record_features import RecordFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


def _make_bout_features(sample_basis: SampleBasis) -> BoutFeatures:
    return BoutFeatures(
        sample_basis=sample_basis,
        features=np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ),
        bout_starts=np.array([10.0, 20.0]),
        bout_ends=np.array([15.0, 25.0]),
        feature_names=[FeatureType.DAY_N, FeatureType.BOUT_DURATION],
        sample_starts=np.array([0.0, 1.0]),
        sample_ends=np.array([0.5, 1.5]),
        units=["", "s"],
    )


class TestFeatureModels(unittest.TestCase):
    def test_bout_features_shape_metadata(self):
        features = _make_bout_features(SampleBasis.EPOCH)
        self.assertEqual(features.num_bouts, 2)
        self.assertEqual(features.num_feature_types, 2)
        self.assertEqual(features.num_samples, 2)
        self.assertEqual(features.features.shape, (2, 2, 2))

    def test_bout_features_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            BoutFeatures(
                sample_basis=SampleBasis.EPOCH,
                features=np.array([[1.0, 2.0], [3.0, 4.0]]),
                bout_starts=np.array([10.0]),
                bout_ends=np.array([15.0]),
                feature_names=[FeatureType.DAY_N],
                sample_starts=np.array([0.0, 1.0]),
                sample_ends=np.array([0.5, 1.5]),
                units=[""],
            )

    def test_record_features_basis_validation(self):
        metadata = FeatureMetadata(
            feature_identifier=FeatureIdentifier("feature_1"),
            user_identifier=UserIdentifier("user_1"),
            imu_data_identifier=IMUDataIdentifier("imu_1"),
        )
        epoch_features = _make_bout_features(SampleBasis.EPOCH)
        stride_features = _make_bout_features(SampleBasis.STRIDE)
        record = RecordFeatures(
            epoch_features=epoch_features,
            stride_features=stride_features,
            feature_metadata=metadata,
        )
        self.assertEqual(record.get_data_id().value, "feature_1")
        self.assertEqual(record.get_associated_data_id().value, "imu_1")
        self.assertIs(record.get_features_by_basis(SampleBasis.EPOCH), epoch_features)
        self.assertIs(record.get_features_by_basis(SampleBasis.STRIDE), stride_features)


if __name__ == "__main__":
    unittest.main()
