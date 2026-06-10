try:
    import skdh
except ModuleNotFoundError:
    skdh = None

from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.processing.imu_validation import ImuValidationMixin


class SkdhPipelineBuilder(ImuValidationMixin):
    GAIT_RES_KEY = "GaitLumbar"

    def build_pipeline(self, profile: ExtractionProfile):
        if skdh is None:
            return None
        pipeline = skdh.Pipeline()
        treadmill_profile = profile == ExtractionProfile.TREADMILL
        if not treadmill_profile:
            pipeline.add(
                self.build_step(
                    skdh.preprocessing.GetDayWindowIndices,
                    bases=[0],
                    periods=[24],
                )
            )
            pipeline.add(self.build_step(skdh.preprocessing.CalibrateAccelerometer))
            pipeline.add(self.build_step(skdh.context.PredictGaitLumbarLgbm))
            pipeline.add(self.build_step(skdh.gait.GaitLumbar, min_bout_time=8.0))
        else:
            pipeline.add(self.build_step(skdh.gait.GaitLumbar, min_bout_time=4.0))
        return pipeline

    @staticmethod
    def library_version() -> str:
        if skdh is None:
            return "unavailable"
        return getattr(skdh, "__version__", "unknown")
