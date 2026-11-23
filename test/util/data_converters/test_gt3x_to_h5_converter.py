import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data_io.import_export.importers.data.imu.imu_data_importer import IMUDataImporter
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.data_converters.imu.gt3x.gt3x_to_h5_converter import (
    GT3XInfo,
    GT3XToH5Converter,
    read_gt3x,
)
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)
from test.util.data_converters.gt3x_fixtures import (
    DATE_OF_BIRTH_TICKS,
    SAMPLE_RATE,
    START_EPOCH_S,
    SUBJECT_NAME,
    counts_for_second,
    make_gt3x,
    make_info,
)

N_SECONDS = 4


class TestGT3XInfo(unittest.TestCase):
    def test_parses_device_and_subject_fields_but_not_identifiers(self):
        info = GT3XInfo.parse(make_info(N_SECONDS))
        self.assertEqual(info.serial_number, "TEST0000001")
        self.assertEqual(info.sample_rate, SAMPLE_RATE)
        self.assertEqual(info.acceleration_scale, 256.0)
        self.assertEqual((info.acceleration_min, info.acceleration_max), (-8.0, 8.0))
        self.assertEqual((info.sex, info.height_cm, info.mass_kg, info.age, info.limb),
                         ("Female", 160.02, 60.5, 74.0, "Back"))
        self.assertEqual(info.instrument_name, "actigraph_wGT3XBT")
        self.assertNotIn(SUBJECT_NAME, repr(info))
        self.assertNotIn(DATE_OF_BIRTH_TICKS, repr(info))

    def test_missing_required_field_raises(self):
        with self.assertRaises(ValueError):
            GT3XInfo.parse("Serial Number: X\nDevice Type: wGT3XBT\n")


class TestReadGt3x(unittest.TestCase):
    def test_reads_counts_as_g_with_timestamps(self):
        recording = read_gt3x(make_gt3x(N_SECONDS))
        self.assertEqual(recording.n_samples, N_SECONDS * SAMPLE_RATE)
        self.assertFalse(recording.repair.repaired)
        expected = np.vstack([counts_for_second(s) for s in range(N_SECONDS)]) / 256.0
        np.testing.assert_array_equal(recording.acceleration_g, expected.astype(np.float32))
        self.assertEqual(recording.acceleration_g.dtype, np.float32)
        self.assertAlmostEqual(recording.time[0], START_EPOCH_S)
        self.assertTrue(np.all(np.diff(recording.time) > 0))
        self.assertAlmostEqual(np.median(np.diff(recording.time)), 1.0 / SAMPLE_RATE, places=5)

    def test_repairs_stray_bytes_transparently(self):
        clean = read_gt3x(make_gt3x(N_SECONDS))
        repaired = read_gt3x(make_gt3x(N_SECONDS, junk=b"\x00\x07\x00"))
        self.assertTrue(repaired.repair.repaired)
        self.assertEqual(repaired.repair.dropped_bytes, 3)
        np.testing.assert_array_equal(repaired.acceleration_g, clean.acceleration_g)
        np.testing.assert_array_equal(repaired.time, clean.time)

    def test_reads_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.gt3x"
            path.write_bytes(make_gt3x(N_SECONDS))
            self.assertEqual(read_gt3x(path).n_samples, N_SECONDS * SAMPLE_RATE)


class TestGT3XToH5Converter(unittest.TestCase):
    def test_axis_mapping(self):
        sensor = GT3XToH5Converter.build_sensor_data(read_gt3x(make_gt3x(N_SECONDS)))
        vertical = sensor.get_data_by_anatomical_axis(AnatomicalCoordinateSystem.VERTICAL)
        self.assertEqual(vertical.sensor_axis.name, SensorCoordinateSystem.Y)
        self.assertGreater(float(np.mean(vertical.data)), 0.9)  # gravity on Y
        self.assertEqual(sensor.metadata.sampling_rate, float(SAMPLE_RATE))
        self.assertEqual(sensor.metadata.unit, "g")

    def test_convert_round_trips_and_carries_no_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            converter = GT3XToH5Converter()
            record_dir = converter.convert(
                make_gt3x(N_SECONDS, junk=b"\x05"),
                Path(tmp),
                IMUDataIdentifier("imu_test"),
                UserIdentifier("FaFRA_999"),
            )
            self.assertEqual(record_dir.name, "imu_data_imu_test")
            imu = IMUDataImporter().import_data(record_dir)
            self.assertEqual(imu.metadata.user_identifier.value, "FaFRA_999")
            self.assertEqual(imu.metadata.instrument_identifier.serial_number, "TEST0000001")
            sensor = imu.data[0]
            self.assertEqual(len(sensor.time), N_SECONDS * SAMPLE_RATE)
            expected_x = np.concatenate([counts_for_second(s)[:, 0] for s in range(N_SECONDS)]) / 256.0
            np.testing.assert_allclose(
                sensor.get_data_by_sensor_axis(SensorCoordinateSystem.X).data, expected_x
            )

            raw = (record_dir / "imu_data.h5").read_bytes()
            self.assertNotIn(SUBJECT_NAME.encode(), raw)
            self.assertNotIn(DATE_OF_BIRTH_TICKS.encode(), raw)
            with h5py.File(record_dir / "imu_data.h5") as h5:
                group = h5["imu_data/sensor_data/accelerometer"]
                self.assertEqual(group["data"].dtype, np.float32)
                self.assertEqual(group["time"].dtype, np.float64)
                self.assertEqual(group["data"].shape, (3, N_SECONDS * SAMPLE_RATE))


if __name__ == "__main__":
    unittest.main()
