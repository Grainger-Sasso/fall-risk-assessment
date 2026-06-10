import unittest

import numpy as np

from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.feature.feature_type import FeatureType
from src.data_types.instrument.sensor_type import SensorType
from src.gait_features.processing.epoch_feature_generator import EpochFeatureGenerator
from src.gait_features.processing.signal_preprocessing import (
    AccelOutputUnit,
    convert_accel,
    infer_accel_unit,
)
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import AnatomicalAxis
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


def _make_sensor_data() -> SensorData:
    sampling_rate_hz = 10.0
    time = np.arange(0.0, 12.0, 1.0 / sampling_rate_hz)
    x = np.sin(2 * np.pi * 1.2 * time)
    y = np.sin(2 * np.pi * 0.8 * time + 0.2)
    z = np.sin(2 * np.pi * 1.0 * time + 0.4)
    data = [
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
    ]
    return SensorData(
        data=data,
        time=time,
        idle_mask=np.zeros_like(time),
        metadata=SensorMetadata(
            sensor_type=SensorType.ACCELEROMETER,
            sampling_rate=sampling_rate_hz,
            unit="m/s^2",
        ),
    )


class TestEpochFeatureGenerator(unittest.TestCase):
    def test_generate_epoch_features_with_overlap(self):
        sensor_data = _make_sensor_data()
        generator = EpochFeatureGenerator(window_seconds=8.0, overlap_seconds=2.0)

        features = generator.build(
            sensor_data=sensor_data,
            bout_sample_ranges=[(0, len(sensor_data.time))],
            bout_time_ranges=[(float(sensor_data.time[0]), float(sensor_data.time[-1]))],
        )

        self.assertEqual(features.features.shape[0], 1)
        self.assertEqual(features.features.shape[1], len(FeatureType.get_epoch_feature_types()))
        self.assertGreaterEqual(features.features.shape[2], 1)
        self.assertEqual(features.sample_starts[0], 0.0)
        if len(features.sample_starts) > 1:
            self.assertAlmostEqual(features.sample_starts[1], 6.0, places=3)
        mean_index = FeatureType.get_epoch_feature_types().index(FeatureType.EPOCH_VERTICAL_MEAN)
        self.assertTrue(np.isfinite(features.features[0, mean_index, 0]))

    def test_accel_mps2_override_scales_constant_g_input(self):
        sampling_rate_hz = 10.0
        time = np.arange(0.0, 12.0, 1.0 / sampling_rate_hz)
        constant_g = 0.5
        data = [
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.ANTEROPOSTERIOR),
                SensorAxis(SensorCoordinateSystem.X),
                np.full_like(time, constant_g),
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.MEDIOLATERAL),
                SensorAxis(SensorCoordinateSystem.Y),
                np.zeros_like(time),
            ),
            UniaxialSensorData(
                AnatomicalAxis(AnatomicalCoordinateSystem.VERTICAL),
                SensorAxis(SensorCoordinateSystem.Z),
                np.zeros_like(time),
            ),
        ]
        sensor_data = SensorData(
            data=data,
            time=time,
            idle_mask=np.zeros_like(time),
            metadata=SensorMetadata(
                sensor_type=SensorType.ACCELEROMETER,
                sampling_rate=sampling_rate_hz,
                unit="g",
            ),
        )
        raw = np.column_stack(
            [
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data,
            ]
        )
        accel_mps2 = convert_accel(raw, infer_accel_unit(raw), AccelOutputUnit.MPS2)
        generator = EpochFeatureGenerator(window_seconds=8.0, overlap_seconds=2.0)
        bout_ranges = [(0, len(sensor_data.time))]
        bout_times = [(float(sensor_data.time[0]), float(sensor_data.time[-1]))]

        raw_features = generator.build(
            sensor_data=sensor_data,
            bout_sample_ranges=bout_ranges,
            bout_time_ranges=bout_times,
        )
        normalized_features = generator.build(
            sensor_data=sensor_data,
            bout_sample_ranges=bout_ranges,
            bout_time_ranges=bout_times,
            accel_mps2=accel_mps2,
        )

        mean_index = FeatureType.get_epoch_feature_types().index(
            FeatureType.EPOCH_ANTEROPOSTERIOR_MEAN
        )
        raw_mean = raw_features.features[0, mean_index, 0]
        normalized_mean = normalized_features.features[0, mean_index, 0]
        self.assertAlmostEqual(raw_mean, constant_g, places=6)
        self.assertAlmostEqual(normalized_mean, constant_g * 9.80665, places=4)


if __name__ == "__main__":
    unittest.main()
