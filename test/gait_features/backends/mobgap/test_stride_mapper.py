import unittest

import numpy as np
import pandas as pd

from src.data_types.feature.mobgap_feature_type import MobgapFeatureType
from src.gait_features.backends.mobgap.stride_mapper import MobgapStrideMapper
from src.gait_features.contracts.extraction_result import BoutSegment


class TestMobgapStrideMapper(unittest.TestCase):
    def test_maps_per_stride_parameters_to_tensor(self):
        per_stride = pd.DataFrame(
            {
                "start": [100, 160],
                "end": [150, 210],
                "lr_label": ["right", "left"],
                "stride_duration_s": [1.0, 1.1],
                "cadence_spm": [100.0, 98.0],
                "stride_length_m": [1.2, 1.15],
                "walking_speed_mps": [0.9, 0.88],
            },
            index=pd.MultiIndex.from_tuples([(0, "0_0"), (0, "0_1")], names=["wb_id", "s_id"]),
        )
        raw_ic = pd.DataFrame(
            {"ic": [100, 160, 220], "lr_label": ["right", "left", "right"]},
            index=pd.MultiIndex.from_product([[0], [0, 1, 2]], names=["gs_id", "step_id"]),
        )
        bout_segments = [
            BoutSegment(
                sample_start=90,
                sample_end=230,
                event_start=0,
                event_end=1,
                start_time=0.9,
                end_time=2.3,
            )
        ]
        mapper = MobgapStrideMapper()
        bout_features = mapper.build_bout_features(
            per_stride_parameters=per_stride,
            raw_ic_list=raw_ic,
            bout_segments=bout_segments,
            sampling_rate_hz=100.0,
        )

        self.assertEqual(bout_features.num_bouts, 1)
        self.assertEqual(len(bout_features.feature_names), len(MobgapFeatureType.get_stride_feature_types()))
        self.assertEqual(bout_features.num_samples, 2)
        cadence_index = bout_features.feature_names.index(MobgapFeatureType.CADENCE_SPM)
        self.assertAlmostEqual(bout_features.features[0, cadence_index, 0], 100.0)
        self.assertAlmostEqual(bout_features.features[0, cadence_index, 1], 98.0)


if __name__ == "__main__":
    unittest.main()
