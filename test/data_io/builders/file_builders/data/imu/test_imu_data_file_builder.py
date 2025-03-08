from src.data_io.builders.file_builders.data.imu.imu_data_file_builder import (
    IMUDataFileBuilder,
)
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_model.data.imu.imu_data import IMUData
from src.data_types.instrument.sensor_type import SensorType
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import IMUDataHelper, TestConstants


class TestIMUDataFileBuilder(BaseTest):
    def setUp(self):
        self.builder = IMUDataFileBuilder()
        self.data_helper = IMUDataHelper()

    def test_build_valid_imu_data(self):
        """Test builder.build() for IMUData -> HDF5Group"""
        # Create test IMU data
        imu_data: IMUData = self.data_helper.create_test_imu_data()
        # Test building HDF5 group
        result: HDF5Group = self.builder.build(imu_data)

        # Assert one group with name imu_data
        self.assertIsInstance(result, HDF5Group)
        self.assertEqual(result.name, IMUDataFields.IMU_DATA.value)
        # Assert parent group attributes
        imu_data_attr = result.attributes
        self.assertIsInstance(imu_data_attr, dict)
        self.assertEqual(len(imu_data_attr.items()), 4)
        self.assertIn(IMUDataFields.IMU_DATA_IDENTIFIER.value, imu_data_attr.keys())
        self.assertIn(IMUDataFields.USER_IDENTIFIER.value, imu_data_attr.keys())
        self.assertIn(IMUDataFields.INSTRUMENT_NAME.value, imu_data_attr.keys())
        self.assertIn(IMUDataFields.SERIAL_NUMBER.value, imu_data_attr.keys())
        self.assertEqual(
            imu_data_attr[IMUDataFields.IMU_DATA_IDENTIFIER.value],
            TestConstants.IMU_DATA_ID.value,
        )
        self.assertEqual(
            imu_data_attr[IMUDataFields.USER_IDENTIFIER.value],
            TestConstants.USER_DATA_ID.value,
        )
        self.assertEqual(
            imu_data_attr[IMUDataFields.INSTRUMENT_NAME.value],
            TestConstants.INSTRUMENT_NAME.value,
        )
        self.assertEqual(
            imu_data_attr[IMUDataFields.SERIAL_NUMBER.value],
            TestConstants.SERIAL_NUMBER.value,
        )
        # Assert single subgroup of sensor data
        self.assertEqual(len(result.items), 1)
        sensor_group = result.items[0]
        self.assertEqual(sensor_group.name, IMUDataFields.SENSOR_DATA.value)
        self.assertEqual(sensor_group.attributes, {})
        # Assert two sensor data subgroups
        self.assertEqual(len(sensor_group.items), len(TestConstants.SENSORS.value))
        self.assertIn(
            IMUDataFields.ACCELEROMETER.value,
            [group.name for group in sensor_group.items],
        )
        self.assertIn(
            IMUDataFields.GYROSCOPE.value,
            [group.name for group in sensor_group.items],
        )
        # Assert for each group, assert time, data, and attributes
        for sub_group in sensor_group.items:
            # Assert subgroup datasets
            self.assertEqual(len(sub_group.items), 2)
            self.assertIn(
                IMUDataFields.TIME.value,
                [group.name for group in sub_group.items],
            )
            self.assertIn(
                IMUDataFields.DATA.value,
                [group.name for group in sub_group.items],
            )
            time = sub_group.get_item_by_name(IMUDataFields.TIME.value)
            self.assertEqual(time.data, TestConstants.TIME_DATA.value)
            self.assertEqual(time.attributes, {})
            sensor_data = sub_group.get_item_by_name(IMUDataFields.DATA.value)
            self.assertEqual(sensor_data.data, TestConstants.IMU_DATA.value)
            self.assertEqual(sensor_data.attributes, {})
            # Assert subgroup attributes
            sub_group_attr = sub_group.attributes
            self.assertIsInstance(sub_group_attr, dict)
            self.assertEqual(len(sub_group_attr.items()), 6)

            self.assertIn(IMUDataFields.SENSOR_TYPE.value, sub_group_attr.keys())
            self.assertIn(
                IMUDataFields.ORIENTATION_MAP_SENSOR.value, sub_group_attr.keys()
            )
            self.assertIn(
                IMUDataFields.ORIENTATION_MAP_ANATOMICAL.value, sub_group_attr.keys()
            )
            self.assertIn(IMUDataFields.SAMPLING_RATE.value, sub_group_attr.keys())
            self.assertIn(IMUDataFields.UNIT.value, sub_group_attr.keys())
            self.assertIn(IMUDataFields.AXIS_NAMES.value, sub_group_attr.keys())

            self.assertIn(
                sub_group_attr[IMUDataFields.SENSOR_TYPE.value],
                [SensorType.ACCELEROMETER.value, SensorType.GYROSCOPE.value],
            )
            self.assertEqual(
                sub_group_attr[IMUDataFields.ORIENTATION_MAP_SENSOR.value],
                TestConstants.ORIENTATION_MAP_SENSOR.value,
            )
            self.assertEqual(
                sub_group_attr[IMUDataFields.ORIENTATION_MAP_ANATOMICAL.value],
                TestConstants.ORIENTATION_MAP_ANATOM.value,
            )
            self.assertEqual(
                sub_group_attr[IMUDataFields.SAMPLING_RATE.value],
                TestConstants.SAMPLING_RATE.value,
            )
            self.assertEqual(
                sub_group_attr[IMUDataFields.UNIT.value],
                TestConstants.UNIT.value,
            )
            self.assertEqual(
                sub_group_attr[IMUDataFields.AXIS_NAMES.value],
                TestConstants.AXIS_NAMES.value,
            )

    def test_build_empty_imu_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestIMUDataFileBuilder.run_tests()
