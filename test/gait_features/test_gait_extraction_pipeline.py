import unittest
from unittest.mock import MagicMock

from src.gait_features.backends.mobgap.backend import MobgapBackend
from src.gait_features.backends.skdh.backend import SkdhBackend
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.pipeline.gait_extraction_pipeline import GaitExtractionPipeline
from src.gait_features.processing.backend_requirements import BackendRequirementsError
from src.gait_features.processing.record_feature_builder import RecordFeatureBuilder
from test.fixtures.synthetic_imu import minimal_skdh_imu_data


class TestGaitExtractionPipeline(unittest.TestCase):
    def test_skdh_pipeline(self):
        pipeline = GaitExtractionPipeline.create(
            backend=GaitExtractionBackendId.SKDH,
            profile=ExtractionProfile.FREE_LIVING,
        )
        self.assertIsInstance(pipeline.extractor, SkdhBackend)
        self.assertIsInstance(pipeline.record_feature_builder, RecordFeatureBuilder)

    def test_mobgap_pipeline(self):
        pipeline = GaitExtractionPipeline.create(
            backend=GaitExtractionBackendId.MOBGAP,
            profile=ExtractionProfile.MOBGAP_HEALTHY,
        )
        self.assertIsInstance(pipeline.extractor, MobgapBackend)
        self.assertIsInstance(pipeline.record_feature_builder, RecordFeatureBuilder)

    def test_incompatible_profile_raises(self):
        with self.assertRaises(ValueError):
            GaitExtractionPipeline.create(
                backend=GaitExtractionBackendId.MOBGAP,
                profile=ExtractionProfile.TREADMILL,
            )

    def test_run_extracts_then_builds_record_features(self):
        extractor = MagicMock()
        extractor.extract.return_value = "extraction_result"
        builder = MagicMock()
        builder.build_from_extraction.return_value = "record_features"
        pipeline = GaitExtractionPipeline(
            backend=GaitExtractionBackendId.SKDH,
            profile=ExtractionProfile.TREADMILL,
            extractor=extractor,
            record_feature_builder=builder,
        )
        imu_data = minimal_skdh_imu_data("imu_1")

        result = pipeline.run(imu_data=imu_data, user_data="user_data", instrument_specifications=None)

        self.assertEqual(result, "record_features")
        extractor.extract.assert_called_once_with(
            imu_data=imu_data,
            user_data="user_data",
            instrument_specifications=None,
            profile=ExtractionProfile.TREADMILL,
        )
        builder.build_from_extraction.assert_called_once_with(
            extraction_result="extraction_result",
            imu_data=imu_data,
            user_data="user_data",
        )

    def test_run_checks_backend_requirements_before_extracting(self):
        extractor = MagicMock()
        pipeline = GaitExtractionPipeline(
            backend=GaitExtractionBackendId.MOBGAP,
            profile=ExtractionProfile.MOBGAP_HEALTHY,
            extractor=extractor,
            record_feature_builder=MagicMock(),
        )

        with self.assertRaises(BackendRequirementsError):
            pipeline.run(
                imu_data=minimal_skdh_imu_data("imu_accel_only"),
                user_data="user_data",
                instrument_specifications=None,
            )
        extractor.extract.assert_not_called()


class TestExtractionProfileResolve(unittest.TestCase):
    def test_explicit_profile_wins(self):
        profile = ExtractionProfile.resolve(
            backend=GaitExtractionBackendId.SKDH,
            explicit_profile=ExtractionProfile.FREE_LIVING,
            treadmill_profile=True,
        )
        self.assertEqual(profile, ExtractionProfile.FREE_LIVING)

    def test_treadmill_flag(self):
        profile = ExtractionProfile.resolve(
            backend=GaitExtractionBackendId.SKDH,
            treadmill_profile=True,
        )
        self.assertEqual(profile, ExtractionProfile.TREADMILL)

    def test_backend_defaults(self):
        self.assertEqual(
            ExtractionProfile.resolve(backend=GaitExtractionBackendId.SKDH),
            ExtractionProfile.FREE_LIVING,
        )
        self.assertEqual(
            ExtractionProfile.resolve(backend=GaitExtractionBackendId.MOBGAP),
            ExtractionProfile.MOBGAP_HEALTHY,
        )


if __name__ == "__main__":
    unittest.main()
