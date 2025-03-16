import numpy as np  # type: ignore

from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import IMUDataHelper, TestConstants


class TestIMUDataBuilder(BaseTest):
    def setUp(self):
        self.builder = IMUDataBuilder()
        self.data_helper = IMUDataHelper()

    def test_build_valid_data(self):
        """Test builder.build() for HDF5Group -> IMUData"""
        # Create test HDF5 group with IMU data
        imu_data_hdf5: HDF5Group = self.data_helper.create_test_imu_data_hdf5()
        # Test building IMU data
        result: IMUData = self.builder.build(imu_data_hdf5)

        # Basic type assertion
        self.assertIsInstance(result, IMUData)

        # Test metadata
        self.assertIsInstance(result.metadata.imu_data_identifier, IMUDataIdentifier)
        self.assertEqual(
            result.metadata.imu_data_identifier.value, TestConstants.IMU_DATA_ID.value
        )
        self.assertIsInstance(result.metadata.user_identifier, UserIdentifier)
        self.assertEqual(
            result.metadata.user_identifier.value, TestConstants.USER_DATA_ID.value
        )
        self.assertIsInstance(
            result.metadata.instrument_identifier, InstrumentIdentifier
        )
        self.assertEqual(
            result.metadata.instrument_identifier.name,
            TestConstants.INSTRUMENT_NAME.value,
        )
        self.assertEqual(
            result.metadata.instrument_identifier.serial_number,
            TestConstants.SERIAL_NUMBER.value,
        )

        # Test timestamps
        self.assertEqual(result.start_time, TestConstants.TIME_DATA.value[0])
        self.assertEqual(result.end_time, TestConstants.TIME_DATA.value[-1])

        # Test epoch data
        self.assertEqual(len(result.data), 1)  # Single epoch
        epoch = result.data[0]
        self.assertIsInstance(epoch, EpochIMUData)

        # Test epoch timestamps
        self.assertEqual(epoch.epoch_start_time, TestConstants.TIME_DATA.value[0])
        self.assertEqual(epoch.epoch_end_time, TestConstants.TIME_DATA.value[-1])

        # Test sensor data for each sensor type
        for sensor_data in result.data[0].data:  # For each sensor in the epoch
            self.assertIn(
                sensor_data.metadata.sensor_type,
                [sensor_type for sensor_type, _ in TestConstants.SENSORS.value],
            )
            self.assertEqual(
                sensor_data.metadata.sampling_rate, TestConstants.SAMPLING_RATE.value
            )
            self.assertEqual(sensor_data.metadata.unit, TestConstants.UNIT.value)

            # Test sensor time data
            np.testing.assert_array_equal(
                sensor_data.time, TestConstants.TIME_DATA.value
            )
            np.testing.assert_array_equal(
                sensor_data.idle_mask, TestConstants.IDLE_MASK.value
            )
            # Test getting data by each axis
            for ix, axis in enumerate(
                [
                    SensorCoordinateSystem.X,
                    SensorCoordinateSystem.Y,
                    SensorCoordinateSystem.Z,
                ]
            ):
                uniaxial_data = sensor_data.get_data_by_sensor_axis(axis)

                # Assert it's the right type
                self.assertIsInstance(uniaxial_data, UniaxialSensorData)

                # Assert correct sensor axis
                self.assertEqual(uniaxial_data.sensor_axis.name, axis)

                # Assert correct anatomical axis mapping
                expected_anatomical_axis = TestConstants.MODEL_ORIENTATION_MAP.value[
                    axis
                ]
                self.assertEqual(
                    uniaxial_data.anatomical_axis.name,
                    expected_anatomical_axis,
                )

                # Assert correct data values - each axis has different test values
                np.testing.assert_array_equal(
                    uniaxial_data.data, np.array(TestConstants.IMU_DATA.value[ix])
                )

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestIMUDataBuilder.run_tests()
