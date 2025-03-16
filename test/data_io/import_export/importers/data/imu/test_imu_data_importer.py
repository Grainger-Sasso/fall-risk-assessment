import json
import tempfile
from pathlib import Path

import numpy as np

from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataFileNames,
    IMUDataImporter,
)
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
)
from src.data_model.data.user.user_data import UserData
from src.data_types.instrument.sensor_type import SensorType
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import IMUDataHelper, TestConstants


class TestIMUDataImporter(BaseTest):
    def setUp(self):
        self.importer = IMUDataImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test HDF5 data using helper
        self.imu_helper = IMUDataHelper()
        self.temp_file = self.temp_path / f"{IMUDataFileNames.IMU_DATA.value}.h5"
        self.imu_helper.create_test_imu_data_file(self.temp_file)

    def tearDown(self):
        if self.temp_file.exists():
            self.temp_file.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Verify result type
        self.assertIsInstance(result, IMUData)

        # Verify metadata
        self.assertEqual(
            result.metadata.imu_data_identifier.value, TestConstants.IMU_DATA_ID.value
        )
        self.assertEqual(
            result.metadata.instrument_identifier.name,
            TestConstants.INSTRUMENT_NAME.value,
        )
        self.assertEqual(
            result.metadata.instrument_identifier.serial_number,
            TestConstants.SERIAL_NUMBER.value,
        )

        # Verify timestamps
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
        for sensor_data in result.data[0].data:
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

                # Assert correct data values
                np.testing.assert_array_equal(
                    uniaxial_data.data, np.array(TestConstants.IMU_DATA.value[ix])
                )

    def test_missing_file(self):
        # Remove the required file
        self.temp_file.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestIMUDataImporter.run_tests()
