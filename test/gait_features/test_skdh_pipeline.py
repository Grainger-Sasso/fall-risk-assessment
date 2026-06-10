import unittest

import skdh

from src.gait_features.backends.skdh.pipeline import SkdhPipelineBuilder
from src.gait_features.config.extraction_profile import ExtractionProfile


class TestSkdhPipeline(unittest.TestCase):
    def test_free_living_gait_lumbar_only_uses_supported_kwargs(self):
        step = SkdhPipelineBuilder().build_step(
            skdh.gait.GaitLumbar,
            min_bout_time=8.0,
            min_bout_duration=8.0,
            min_steps=6,
        )
        self.assertEqual(step.min_bout_time, (8.0,))
        self.assertFalse(hasattr(step, "min_bout_duration"))
        self.assertFalse(hasattr(step, "min_steps"))

    def test_treadmill_pipeline_sets_shorter_min_bout_time(self):
        pipeline = SkdhPipelineBuilder().build_pipeline(ExtractionProfile.TREADMILL)
        self.assertIsNotNone(pipeline)
        gait_step = list(pipeline)[-1]
        self.assertEqual(gait_step.min_bout_time, (4.0,))

    def test_free_living_pipeline_uses_default_min_bout_time(self):
        pipeline = SkdhPipelineBuilder().build_pipeline(ExtractionProfile.FREE_LIVING)
        gait_step = list(pipeline)[-1]
        self.assertEqual(gait_step.min_bout_time, (8.0,))


if __name__ == "__main__":
    unittest.main()
