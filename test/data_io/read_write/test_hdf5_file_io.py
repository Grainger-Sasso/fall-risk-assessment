import tempfile
from pathlib import Path

import h5py
import numpy as np  # type: ignore

from src.data_io.builders.file_builders.data.imu.imu_data_file_builder import (
    IMUDataFileBuilder,
)
from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import (
    FileIOHelper,
    IMUDataHelper,
    TestConstants,
)


class TestHDF5FileIO(BaseTest):
    def setUp(self):
        self.reader = HDF5FileReader()
        self.writer = HDF5FileWriter()
        self.helper = FileIOHelper()
        self.builder = IMUDataBuilder()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir) / "test.h5"

    def tearDown(self):
        if self.temp_path.exists():
            self.temp_path.unlink()
        self.temp_path.parent.rmdir()

    def test_write_and_read_hdf5(self):
        # Create test data
        test_data: HDF5Group = self.helper.create_test_hdf5_group()

        # Write data
        success, error = self.writer.write(self.temp_path, test_data)
        self.assertTrue(success, f"Failed to write HDF5: {error}")
        self.assertTrue(self.temp_path.exists())

        # Read data back
        result: HDF5Group = self.reader.read(self.temp_path)

        # Verify data
        self.assertIsInstance(result, HDF5Group)
        self.assertEqual(result.name, test_data.name)
        self.assertEqual(result.attributes, test_data.attributes)

        # Verify dataset
        self.assertEqual(len(result.items), len(test_data.items))
        self.assertIsInstance(result.items, list)
        for ix, result_dataset in enumerate(result.items):
            self.assertIsInstance(result_dataset, HDF5Dataset)
            test_dataset = test_data.items[ix]
            self.assertEqual(result_dataset.name, test_dataset.name)
            self.assertEqual(result_dataset.attributes, test_dataset.attributes)
            self.assertEqual(result_dataset.data, test_dataset.data)

    def test_read_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.reader.read(Path("nonexistent.h5"))

    def test_read_invalid_extension(self):
        invalid_path = Path(self.temp_dir) / "test.txt"
        with self.assertRaises(ValueError):
            self.reader.read(invalid_path)

    def test_write_and_read_imu_hdf5(self):
        """Test writing and reading complex IMU data HDF5 structure."""
        # Create test IMU data HDF5 group
        imu_helper = IMUDataHelper()
        test_data: HDF5Group = imu_helper.create_test_imu_data_hdf5()

        # Write data
        success, error = self.writer.write(self.temp_path, test_data)
        self.assertTrue(success, f"Failed to write IMU HDF5: {error}")
        self.assertTrue(self.temp_path.exists())

        # Read data back
        result: HDF5Group = self.reader.read(self.temp_path)

        # Verify top-level group
        self.assertIsInstance(result, HDF5Group)
        self.assertEqual(result.name, test_data.name)
        self.assertEqual(result.attributes, test_data.attributes)
        self.assertEqual(len(result.items), len(test_data.items))
        sensor_group: HDF5Group = result.get_item_by_name(
            IMUDataFields.SENSOR_DATA.value
        )

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
            np.testing.assert_array_equal(time.data, TestConstants.TIME_DATA.value)
            self.assertEqual(time.attributes, {})
            sensor_data = sub_group.get_item_by_name(IMUDataFields.DATA.value)
            np.testing.assert_array_equal(
                sensor_data.data, TestConstants.IMU_DATA.value
            )
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
            np.testing.assert_array_equal(
                sub_group_attr[IMUDataFields.ORIENTATION_MAP_SENSOR.value],
                TestConstants.ORIENTATION_MAP_SENSOR.value,
            )
            np.testing.assert_array_equal(
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
            np.testing.assert_array_equal(
                sub_group_attr[IMUDataFields.AXIS_NAMES.value],
                TestConstants.AXIS_NAMES.value,
            )

        # Check that the group read in will build model indices correctly
        result_2: IMUData = self.builder.build(result)

        # Basic type assertion
        self.assertIsInstance(result_2, IMUData)

        # Test metadata
        self.assertEqual(
            result_2.metadata.imu_data_identifier.value, TestConstants.IMU_DATA_ID.value
        )
        self.assertEqual(
            result_2.metadata.instrument_identifier.name,
            TestConstants.INSTRUMENT_NAME.value,
        )
        self.assertEqual(
            result_2.metadata.instrument_identifier.serial_number,
            TestConstants.SERIAL_NUMBER.value,
        )

        # Test timestamps
        self.assertEqual(result_2.start_time, TestConstants.TIME_DATA.value[0])
        self.assertEqual(result_2.end_time, TestConstants.TIME_DATA.value[-1])

        # Test epoch data
        self.assertEqual(len(result_2.data), 1)  # Single epoch
        epoch = result_2.data[0]
        self.assertIsInstance(epoch, EpochIMUData)

        # Test epoch timestamps
        self.assertEqual(epoch.epoch_start_time, TestConstants.TIME_DATA.value[0])
        self.assertEqual(epoch.epoch_end_time, TestConstants.TIME_DATA.value[-1])

        # Test sensor data for each sensor type
        for sensor_data in result_2.data[0].data:  # For each sensor in the epoch
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


if __name__ == "__main__":
    TestHDF5FileIO.run_tests()
