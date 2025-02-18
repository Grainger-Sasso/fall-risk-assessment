import unittest

from src.data_io.builders.file_builders.data.imu.imu_data_file_builder import (
    IMUDataFileBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.data.imu.imu_data import IMUData


class TestIMUDataFileBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = IMUDataFileBuilder()

    def test_build_valid_imu_data(self):
        # TODO: Create test IMU data
        imu_data = IMUData([], None, 0.0, 1.0)

        # Test building HDF5 group
        result = self.builder.build(imu_data)

        # Assertions
        self.assertIsInstance(result, HDF5Group)
        # TODO: Add more specific assertions

    def test_build_empty_imu_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    unittest.main()
