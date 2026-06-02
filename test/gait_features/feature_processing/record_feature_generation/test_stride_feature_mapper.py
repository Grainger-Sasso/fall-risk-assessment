import unittest

import numpy as np

from src.data_types.feature.feature_type import FeatureType
from src.gait_features.gait_feature_module import (
    GaitResults,
    StrideFeatureMapper,
)


class TestStrideFeatureMapper(unittest.TestCase):
    def test_build_stride_tensor_from_event_ranges(self):
        gait_results = GaitResults(
            {
                FeatureType.STRIDE_TIME.value: np.array([0.9, 1.1, 1.0, 1.2]),
                FeatureType.GAIT_SPEED.value: np.array([1.3, 1.4, 1.2, 1.1]),
            }
        )
        mapper = StrideFeatureMapper()

        bout_features = mapper.build(
            gait_results=gait_results,
            bout_event_ranges=[(0, 2), (2, 4)],
            bout_time_ranges=[(10.0, 20.0), (30.0, 40.0)],
        )

        self.assertEqual(bout_features.features.shape[0], 2)
        self.assertEqual(bout_features.features.shape[2], 2)
        stride_time_index = mapper.feature_types.index(FeatureType.STRIDE_TIME)
        np.testing.assert_allclose(
            bout_features.features[0, stride_time_index, :],
            np.array([0.9, 1.1]),
        )
        np.testing.assert_allclose(
            bout_features.features[1, stride_time_index, :],
            np.array([1.0, 1.2]),
        )


if __name__ == "__main__":
    unittest.main()
