import unittest

import numpy as np

from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.data.imu.imu_data import IMUData


class TestIMUDataBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = IMUDataBuilder()

    def test_build_valid_data(self):
        # TODO: Create test HDF5 group with IMU data
        input_group = HDF5Group()

        # Test building IMU data
        result = self.builder.build(input_group)

        # Assertions
        self.assertIsInstance(result, IMUData)
        # TODO: Add more specific assertions

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    unittest.main()
