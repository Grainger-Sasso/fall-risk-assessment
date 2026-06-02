import unittest
from typing import Optional
from unittest.mock import MagicMock

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
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.data_model.instrument_specifications.sensor_specifications import SensorSpecification
from src.data_types.instrument.sensor_type import SensorType
from src.gait_features.gait_feature_module import GaitFeatureExtractor
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
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


def _make_imu_data(unit: str, sampling_rate: float) -> IMUData:
    time = np.arange(0.0, 12.0, 0.01)  # 100 Hz
    x = np.sin(2 * np.pi * 1.0 * time)
    y = np.sin(2 * np.pi * 1.2 * time + 0.1)
    z = np.sin(2 * np.pi * 0.8 * time + 0.2)
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
            sampling_rate=sampling_rate,
            unit=unit,
        ),
    )
    metadata = IMUMetadata(
        imu_data_identifier=IMUDataIdentifier("imu_1"),
        user_identifier=UserIdentifier("user_1"),
        instrument_identifier=InstrumentIdentifier("test_imu", "001"),
    )
    return IMUData(
        data=[sensor], metadata=metadata, start_time=float(time[0]), end_time=float(time[-1])
    )


def _make_spec(
    units: Optional[str],
    sampling_rate: Optional[float],
    rng: Optional[tuple[float, float]],
    sensitivity: Optional[float] = 2048.0,
    physical_size: Optional[tuple[Optional[float], Optional[float], Optional[float]]] = (
        1.0,
        1.0,
        1.0,
    ),
) -> IMUSpecifications:
    return IMUSpecifications(
        sensor_specifications=[
            SensorSpecification(
                sensor_type=SensorType.ACCELEROMETER,
                sensor_name="accel",
                units=units,
                range=rng,
                sensitivity=sensitivity,
                resolution=16,
                sampling_rate=sampling_rate,
                noise_density=0.001,
                bias_stability=0.1,
                alignment_error=0.1,
                cross_axis_sensitivity=1.0,
                power_consumption=0.5,
                operating_conditions={"temp": "ok"},
                physical_size=physical_size,
                mass=1.0,
            )
        ],
        specification_id=InstrumentSpecificationIdentifier("spec_1"),
        imu_name="test_imu",
    )


class TestGaitFeatureExtractorMetadataValidation(unittest.TestCase):
    def _build_extractor(self) -> GaitFeatureExtractor:
        extractor = GaitFeatureExtractor(treadmill_profile=True)
        extractor.pipeline = MagicMock(return_value=None)
        extractor.pipeline.run = MagicMock(return_value={"GaitLumbar": {}})
        return extractor

    def test_raises_on_sensor_metadata_unit_mismatch(self):
        extractor = self._build_extractor()
        imu_data = _make_imu_data(unit="m/s^2", sampling_rate=100.0)
        with self.assertRaises(ValueError):
            extractor.extract_gait_features(imu_data=imu_data, user_data=_make_user_data())

    def test_raises_on_sensor_metadata_sampling_rate_mismatch(self):
        extractor = self._build_extractor()
        imu_data = _make_imu_data(unit="g", sampling_rate=50.0)
        with self.assertRaises(ValueError):
            extractor.extract_gait_features(imu_data=imu_data, user_data=_make_user_data())

    def test_raises_on_instrument_spec_range_mismatch(self):
        extractor = self._build_extractor()
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        spec = _make_spec(units="g", sampling_rate=100.0, rng=(-0.2, 0.2))
        with self.assertRaises(ValueError):
            extractor.extract_gait_features(
                imu_data=imu_data,
                user_data=_make_user_data(),
                instrument_specifications=spec,
            )

    def test_none_instrument_spec_values_warn_but_do_not_raise(self):
        extractor = self._build_extractor()
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        spec = _make_spec(
            units="g",
            sampling_rate=100.0,
            rng=(-2.0, 2.0),
            sensitivity=None,
        )
        result = extractor.extract_gait_features(
            imu_data=imu_data,
            user_data=_make_user_data(),
            instrument_specifications=spec,
        )
        self.assertEqual(result.data, {})

    def test_none_physical_size_warns_but_does_not_raise(self):
        extractor = self._build_extractor()
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        spec = _make_spec(
            units="g",
            sampling_rate=100.0,
            rng=(-2.0, 2.0),
            physical_size=None,
        )
        result = extractor.extract_gait_features(
            imu_data=imu_data,
            user_data=_make_user_data(),
            instrument_specifications=spec,
        )
        self.assertEqual(result.data, {})

    def test_none_physical_size_component_warns_but_does_not_raise(self):
        extractor = self._build_extractor()
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        spec = _make_spec(
            units="g",
            sampling_rate=100.0,
            rng=(-2.0, 2.0),
            physical_size=(1.0, None, 1.0),
        )
        result = extractor.extract_gait_features(
            imu_data=imu_data,
            user_data=_make_user_data(),
            instrument_specifications=spec,
        )
        self.assertEqual(result.data, {})

    def test_invalid_non_null_physical_size_component_raises(self):
        extractor = self._build_extractor()
        imu_data = _make_imu_data(unit="g", sampling_rate=100.0)
        spec = _make_spec(
            units="g",
            sampling_rate=100.0,
            rng=(-2.0, 2.0),
            physical_size=(1.0, "bad_value", 1.0),
        )
        with self.assertRaises(ValueError):
            extractor.extract_gait_features(
                imu_data=imu_data,
                user_data=_make_user_data(),
                instrument_specifications=spec,
            )


if __name__ == "__main__":
    unittest.main()
