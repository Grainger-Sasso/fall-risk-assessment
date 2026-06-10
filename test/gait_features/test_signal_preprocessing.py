import unittest
from unittest.mock import MagicMock

import numpy as np

from src.gait_features.backends.skdh.backend import GaitFeatureExtractor
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.processing.signal_preprocessing import (
    AccelOutputUnit,
    BackendSignalProfile,
    GyroOutputUnit,
    convert_accel,
    convert_gyro,
    infer_accel_unit,
)
from src.gait_features.processing.imu_validation import ImuValidationMixin
from test.gait_features.test_gait_feature_extractor_metadata_validation import (
    _make_imu_data,
    _make_spec,
    _make_user_data,
)


class TestSignalPreprocessing(unittest.TestCase):
    def test_backend_profiles(self):
        skdh = BackendSignalProfile.for_backend(GaitExtractionBackendId.SKDH)
        mobgap = BackendSignalProfile.for_backend(GaitExtractionBackendId.MOBGAP)
        self.assertEqual(skdh.accel_unit, AccelOutputUnit.G)
        self.assertEqual(skdh.gyro_unit, GyroOutputUnit.RAD_S)
        self.assertEqual(mobgap.accel_unit, AccelOutputUnit.MPS2)
        self.assertEqual(mobgap.gyro_unit, GyroOutputUnit.DEG_S)

    def test_infer_accel_unit(self):
        g_accel = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.9]])
        ms2_accel = g_accel * 9.80665
        self.assertEqual(infer_accel_unit(g_accel), "g")
        self.assertEqual(infer_accel_unit(ms2_accel), "m/s2")

    def test_convert_accel_g_to_mps2_and_back(self):
        g_accel = np.array([[0.0, 0.0, 1.0]])
        ms2_accel = convert_accel(g_accel, "g", AccelOutputUnit.MPS2)
        self.assertAlmostEqual(ms2_accel[0, 2], 9.80665)
        roundtrip = convert_accel(ms2_accel, "m/s2", AccelOutputUnit.G)
        self.assertAlmostEqual(roundtrip[0, 2], 1.0)

    def test_convert_gyro_deg_to_rad_and_back(self):
        deg = np.array([[90.0, 0.0, 0.0]])
        rad = convert_gyro(deg, "deg/s", GyroOutputUnit.RAD_S)
        self.assertAlmostEqual(rad[0, 0], np.pi / 2.0)
        roundtrip = convert_gyro(rad, "rad/s", GyroOutputUnit.DEG_S)
        self.assertAlmostEqual(roundtrip[0, 0], 90.0)


class TestBackendAccelPreparation(unittest.TestCase):
    def test_skdh_prepare_keeps_g_units(self):
        mixin = ImuValidationMixin()
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        _, accel, _ = mixin.prepare_accelerometer_for_backend(
            imu_data=imu_data,
            instrument_specifications=None,
            treadmill_profile=True,
            backend=GaitExtractionBackendId.SKDH,
        )
        median_norm = float(np.nanmedian(np.linalg.norm(accel, axis=1)))
        self.assertLess(median_norm, 2.5)

    def test_mobgap_prepare_converts_g_to_mps2(self):
        mixin = ImuValidationMixin()
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        _, accel, _ = mixin.prepare_accelerometer_for_backend(
            imu_data=imu_data,
            instrument_specifications=None,
            treadmill_profile=False,
            backend=GaitExtractionBackendId.MOBGAP,
        )
        median_norm = float(np.nanmedian(np.linalg.norm(accel, axis=1)))
        self.assertGreater(median_norm, 5.0)

    def test_skdh_backend_passes_g_scale_accel_to_pipeline(self):
        extractor = GaitFeatureExtractor(treadmill_profile=True)
        extractor.pipeline = MagicMock()
        extractor.pipeline.run = MagicMock(return_value={"GaitLumbar": {}})
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        spec = _make_spec(units="g", sampling_rate=100.0, rng=(-2.0, 2.0))
        extractor.extract_gait_features(
            imu_data=imu_data,
            user_data=_make_user_data(),
            instrument_specifications=spec,
        )
        accel = extractor.pipeline.run.call_args.kwargs["accel"]
        median_norm = float(np.nanmedian(np.linalg.norm(accel, axis=1)))
        self.assertLess(median_norm, 2.5)


if __name__ == "__main__":
    unittest.main()
