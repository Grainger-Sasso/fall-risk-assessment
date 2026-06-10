import unittest

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_model.features.bout_sample_layout import (
    count_usable_samples,
    flatten_bout_features,
    select_usable_samples,
    usable_sample_mask,
)
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis


class TestBoutSampleLayout(unittest.TestCase):
    def test_count_usable_samples_ignores_padding(self):
        features = np.array(
            [
                [[1.0, 2.0, np.nan, np.nan], [3.0, 4.0, np.nan, np.nan]],
                [[5.0, np.nan, np.nan, np.nan], [6.0, np.nan, np.nan, np.nan]],
            ]
        )
        self.assertEqual(count_usable_samples(features), 3)
        flattened = flatten_bout_features(features)
        self.assertEqual(flattened.shape, (8, 2))
        self.assertEqual(select_usable_samples(flattened).shape[0], 3)

    def test_bout_features_properties(self):
        bout = BoutFeatures(
            sample_basis=SampleBasis.STRIDE,
            features=np.array([[[1.0, np.nan], [2.0, 3.0]]]),
            bout_starts=np.array([0.0]),
            bout_ends=np.array([1.0]),
            feature_names=[FeatureType.GAIT_SPEED, FeatureType.CADENCE],
            sample_starts=np.array([0.0, 1.0]),
            sample_ends=np.array([1.0, 2.0]),
            units=["m/s", "steps/min"],
        )
        self.assertEqual(bout.usable_sample_count, 2)
        self.assertEqual(bout.usable_sample_mask().tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
