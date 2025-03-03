import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
    IMUDataFileNames,
)
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


class TestIMUDataFileExporter(BaseTest):
    def setUp(self):
        # Create test objects
        self.exporter = IMUDataFileExporter()
        self.importer = IMUDataImporter()
        self.helper = IMUDataHelper()

        # Create temp directory for test
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test data
        self.test_data = self.helper.create_test_imu_data()

    def tearDown(self):
        # Clean up test directory
        if self.temp_path.exists():
            # Walk directory tree bottom-up to properly remove all contents
            for root, dirs, files in os.walk(self.temp_path, topdown=False):
                # First remove all files in current directory
                for name in files:
                    (Path(root) / name).unlink()
                # Then remove the directory itself
                for name in dirs:
                    (Path(root) / name).rmdir()
            # Finally remove the temp directory
            self.temp_path.rmdir()

    def test_export_data(self):
        # Export the test data
        success, message = self.exporter.export_data(self.temp_path, self.test_data)

        # Verify export succeeded
        self.assertTrue(success, f"Export failed with message: {message}")

        # Verify subdirectory was created with correct name
        expected_subdir = self.temp_path / f"imu_data_{TestConstants.IMU_DATA_ID.value}"
        self.assertTrue(expected_subdir.exists(), "Subdirectory not created")
        self.assertTrue(expected_subdir.is_dir(), "Subdirectory is not a directory")

        # Verify file was created with correct name
        expected_file = expected_subdir / f"{IMUDataFileNames.IMU_DATA.value}.h5"
        self.assertTrue(expected_file.exists(), "Output file not created")
        self.assertTrue(expected_file.is_file(), "Output is not a file")

        # Import the exported data
        result = self.importer.import_data(expected_subdir)

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

    def test_export_data_directory_creation_fails(self):
        # Create a file with the same name as our intended subdirectory to cause creation to fail
        conflict_path = self.temp_path / f"imu_data_{TestConstants.IMU_DATA_ID.value}"
        conflict_path.touch()  # Create file

        # Attempt export
        success, message = self.exporter.export_data(self.temp_path, self.test_data)

        # Verify export failed
        self.assertFalse(success)
        self.assertIn("Failed to create directory", message)

        # Clean up
        conflict_path.unlink()

    def test_export_data_write_fails(self):
        # Create directory structure first with write permissions
        readonly_dir = self.temp_path / "readonly"
        target_dir = readonly_dir / f"imu_data_{TestConstants.IMU_DATA_ID.value}"
        target_dir.mkdir(parents=True)

        # Then make parent directory read-only
        readonly_dir.chmod(0o444)  # Read-only permissions

        # Attempt export
        success, message = self.exporter.export_data(readonly_dir, self.test_data)

        # Verify export failed
        self.assertFalse(success)
        self.assertIn("Permission denied", message)  # More generic error check

        # Clean up - restore permissions to allow deletion
        readonly_dir.chmod(0o777)


if __name__ == "__main__":
    TestIMUDataFileExporter.run_tests()
