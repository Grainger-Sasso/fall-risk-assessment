import unittest

from src.gait_features.build_extraction_pipeline import build_extraction_pipeline
from src.gait_features.backends.mobgap.backend import MobgapBackend
from src.gait_features.backends.skdh.backend import SkdhBackend
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.processing.record_feature_builder import RecordFeatureBuilder


class TestBuildExtractionPipeline(unittest.TestCase):
    def test_skdh_pipeline(self):
        extractor, builder = build_extraction_pipeline(
            backend=GaitExtractionBackendId.SKDH,
            profile=ExtractionProfile.FREE_LIVING,
        )
        self.assertIsInstance(extractor, SkdhBackend)
        self.assertIsInstance(builder, RecordFeatureBuilder)

    def test_mobgap_pipeline(self):
        extractor, builder = build_extraction_pipeline(
            backend=GaitExtractionBackendId.MOBGAP,
            profile=ExtractionProfile.MOBGAP_HEALTHY,
        )
        self.assertIsInstance(extractor, MobgapBackend)
        self.assertIsInstance(builder, RecordFeatureBuilder)

    def test_incompatible_profile_raises(self):
        with self.assertRaises(ValueError):
            build_extraction_pipeline(
                backend=GaitExtractionBackendId.MOBGAP,
                profile=ExtractionProfile.TREADMILL,
            )


if __name__ == "__main__":
    unittest.main()
