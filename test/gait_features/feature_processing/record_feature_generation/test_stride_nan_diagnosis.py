import unittest

import numpy as np

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_model.data.user.clinical.clinical_assessment import ClinicalAssessment
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
    FallerStatus,
    Sex,
)
from src.data_model.data.user.user_data import UserData
from src.data_types.feature.feature_type import FeatureType
from src.data_types.instrument.sensor_type import SensorType
from src.gait_features.gait_feature_module import (
    BoutRange,
    GaitResults,
    RecordFeatureGenerationBuilder,
    StrideFeatureGenerationError,
    StrideFeatureMapper,
    diagnose_stride_nan_source,
)
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.user.clinical_identifier import ClinicalIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import AnatomicalAxis
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)
from src.util.mechanics.units.si.kilogram import Kilogram
from src.util.mechanics.units.si.meter import Meter


def _make_imu_data() -> IMUData:
    sampling_rate_hz = 10.0
    time = np.arange(0.0, 12.0, 1.0 / sampling_rate_hz)
    x = np.sin(2 * np.pi * 1.1 * time)
    y = np.sin(2 * np.pi * 0.9 * time + 0.1)
    z = np.sin(2 * np.pi * 1.0 * time + 0.2)
    sensor = SensorData(
        data=[
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.ANTEROPOSTERIOR),
                SensorAxis(SensorCoordinateSystem.X),
                x,
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.MEDIOLATERAL),
                SensorAxis(SensorCoordinateSystem.Y),
                y,
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.VERTICAL),
                SensorAxis(SensorCoordinateSystem.Z),
                z,
            ),
        ],
        time=time,
        idle_mask=np.zeros_like(time),
        metadata=SensorMetadata(
            sensor_type=SensorType.ACCELEROMETER,
            sampling_rate=sampling_rate_hz,
            unit="m/s^2",
        ),
    )
    metadata = IMUMetadata(
        imu_data_identifier=IMUDataIdentifier("imu_1"),
        user_identifier=UserIdentifier("user_1"),
        instrument_identifier=InstrumentIdentifier("test_imu", "001"),
    )
    return IMUData(data=[sensor], metadata=metadata, start_time=float(time[0]), end_time=float(time[-1]))


def _make_user_data() -> UserData:
    return UserData(
        user_identifier=UserIdentifier("user_1"),
        clinical_demographic_data=ClinicalDemographicData(
            name="Test",
            age=70.0,
            sex=Sex.MALE,
            weight=Kilogram(70.0),
            height=Meter(1.75),
            clinical_identifier=ClinicalIdentifier("clinical_1"),
            faller_status=FallerStatus.NON_FALLER,
        ),
        clinical_assessments=[ClinicalAssessment()],
    )


class TestStrideNaNDiagnosis(unittest.TestCase):
    """Pins down the exact stage at which stride values become NaN/absent."""

    def test_missing_skdh_stride_keys_is_the_nan_source(self):
        # Mirrors the real DB failure: SKDH yielded no stride keys and a
        # zero-width event range, which the mapper fills entirely with NaN.
        gait_results = GaitResults({})
        mapper = StrideFeatureMapper()
        stride_features = mapper.build(
            gait_results=gait_results,
            bout_event_ranges=[(0, 0)],
            bout_time_ranges=[(10.0, 20.0)],
        )
        self.assertTrue(np.isnan(stride_features.features).all())

        report = diagnose_stride_nan_source(
            gait_results=gait_results,
            bout_ranges=[
                BoutRange(
                    sample_start=0,
                    sample_end=100,
                    event_start=0,
                    event_end=0,
                    start_time=10.0,
                    end_time=20.0,
                )
            ],
            stride_features=stride_features,
            stride_feature_types=mapper.feature_types,
        )
        self.assertEqual(report["stage"], "skdh_missing_all_stride_keys")
        self.assertEqual(len(report["present_stride_keys"]), 0)
        self.assertTrue(report["tensor_all_nan"])

    def test_present_stride_keys_yield_ok_diagnosis(self):
        mapper = StrideFeatureMapper()
        gait_results = GaitResults(
            {FeatureType.STRIDE_TIME.value: np.array([0.9, 1.1])}
        )
        stride_features = mapper.build(
            gait_results=gait_results,
            bout_event_ranges=[(0, 2)],
            bout_time_ranges=[(10.0, 20.0)],
        )
        report = diagnose_stride_nan_source(
            gait_results=gait_results,
            bout_ranges=[
                BoutRange(
                    sample_start=0,
                    sample_end=100,
                    event_start=0,
                    event_end=2,
                    start_time=10.0,
                    end_time=20.0,
                )
            ],
            stride_features=stride_features,
            stride_feature_types=mapper.feature_types,
        )
        self.assertEqual(report["stage"], "ok")
        self.assertFalse(report["tensor_all_nan"])

    def test_builder_raises_stride_error_when_no_bouts_detected(self):
        builder = RecordFeatureGenerationBuilder(window_seconds=8.0, overlap_seconds=2.0)
        gait_results = GaitResults(
            {
                "Day N": np.array([]),
                "Bout N": np.array([]),
                "IC Time": np.array([]),
            }
        )
        with self.assertRaises(StrideFeatureGenerationError) as ctx:
            builder.build(
                gait_results=gait_results,
                imu_data=_make_imu_data(),
                user_data=_make_user_data(),
                treadmill_profile=True,
            )
        self.assertEqual(ctx.exception.diagnosis["stage"], "skdh_missing_all_stride_keys")


if __name__ == "__main__":
    unittest.main()
