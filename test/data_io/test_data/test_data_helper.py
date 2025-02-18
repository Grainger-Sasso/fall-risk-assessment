import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
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


class TestDataHelper:
    """Helper class for managing test data files and HDF5 operations.

    This class provides utilities for creating and managing HDF5 test files,
    including standard test data structures and file operations.

    Attributes:
        TEMP_TEST_DATA_DIR (Path): Directory for temporary test files
        writer (JHDF5FileWriter): HDF5 file writer instance
    """

    # Directory for temporary test files
    TEMP_TEST_DATA_DIR = Path(__file__).parent / "temp"

    def __init__(self):
        """Initialize the test data helper."""
        self.writer = HDF5FileWriter()
        self._ensure_temp_dir_exist()

    def _ensure_temp_dir_exist(self) -> None:
        """Ensure temp directory exist."""
        # Create temp directory
        self.TEMP_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def create_test_file(
        self, filename: str, group: Optional[HDF5Group] = None
    ) -> Path:
        """Create a new test HDF5 file in temp directory.

        Args:
            filename (str): Name of the test file
            group (Optional[HDF5Group]): Group to write to file

        Returns:
            Path: Path to created test file

        Raises:
            ValueError: If file creation fails
        """
        if not filename.endswith((".h5", ".hdf5")):
            filename += ".h5"

        file_path = self.TEMP_TEST_DATA_DIR / filename

        if group is None:
            group = HDF5Group(name="root", items=[], attributes={})

        success, error = self.writer.write(file_path, group)
        if not success:
            print("\nFull error from writer:")
            raise ValueError(f"Failed to create test file: {error}")

        return file_path

    def cleanup_test_files(self) -> None:
        """Remove all temporary test files."""
        if self.TEMP_TEST_DATA_DIR.exists():
            shutil.rmtree(self.TEMP_TEST_DATA_DIR)
        self.TEMP_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


class IMUDataHelper:
    """######## IMU DATA ########"""

    def __init__(self):
        self.test_data_helper = TestDataHelper()
        self.constants = {
            "imu_data_id": "test_imu_data_identifier",
            "instrument_name": "testInstrumentName",
            "serial_number": "1234567890",
            "sensors": [
                (SensorType.ACCELEROMETER, IMUDataFields.ACCELEROMETER),
                (SensorType.GYROSCOPE, IMUDataFields.GYROSCOPE),
            ],
            "orientation_map_sensor": [
                IMUDataFields.SENSOR_AXIS_X.value,
                IMUDataFields.SENSOR_AXIS_Y.value,
                IMUDataFields.SENSOR_AXIS_Z.value,
            ],
            "orientation_map_anatom": [
                IMUDataFields.ANATOMICAL_AXIS_ANTEROPOSTERIOR.value,
                IMUDataFields.ANATOMICAL_AXIS_MEDIOLATERAL.value,
                IMUDataFields.ANATOMICAL_AXIS_VERTICAL.value,
            ],
            "sampling_rate": 100.0,
            "unit": "test_unit",
            "axis_names": [
                IMUDataFields.SENSOR_AXIS_X.value,
                IMUDataFields.SENSOR_AXIS_Y.value,
                IMUDataFields.SENSOR_AXIS_Z.value,
            ],
            "time_data": [float(i) / 10 for i in range(0, 11, 1)],
            "imu_data": [[0.0 for i in range(10)] for j in range(3)],
            "model_axis_names": [
                SensorCoordinateSystem.X,
                SensorCoordinateSystem.Y,
                SensorCoordinateSystem.Z,
            ],
            "model_orientation_map": {
                SensorCoordinateSystem.X: AnatomicalCoordinateSystem.ANTEROPOSTERIOR,
                SensorCoordinateSystem.Y: AnatomicalCoordinateSystem.MEDIOLATERAL,
                SensorCoordinateSystem.Z: AnatomicalCoordinateSystem.VERTICAL,
            },
        }

    def create_test_imu_data_file(self) -> Path:
        """Create a test IMU data file.

        Returns:
            Path: Path to created test file
        """
        imu_data_group = HDF5Group(
            name=IMUDataFields.IMU_DATA.value,
            attributes=self._build_test_imu_data_attributes(),
            items=self._build_sensor_data_group(),
        )
        return self.test_data_helper.create_test_file(
            "test_imu_data.h5", imu_data_group
        )

    def _build_test_imu_data_attributes(self) -> Dict[str, Any]:
        """Build test IMU data attributes."""
        test_attributes = {}
        test_attributes[IMUDataFields.IMU_DATA_IDENTIFIER.value] = self.constants[
            "imu_data_id"
        ]
        test_attributes[IMUDataFields.INSTRUMENT_NAME.value] = self.constants[
            "instrument_name"
        ]
        test_attributes[IMUDataFields.SERIAL_NUMBER.value] = self.constants[
            "serial_number"
        ]
        return test_attributes

    def _build_sensor_data_group(self) -> List[HDF5Group]:
        """Build test IMU data items."""
        sensor_data_group = HDF5Group(
            name=IMUDataFields.SENSOR_DATA.value, items=[], attributes={}
        )
        items = []
        for sensor_type, sensor_name in self.constants["sensors"]:
            items.append(self._build_sensor_data_group_item(sensor_type, sensor_name))
        sensor_data_group.items = items
        return [sensor_data_group]

    def _build_sensor_data_group_item(
        self, sensor_type: SensorType, sensor_name: IMUDataFields
    ) -> HDF5Group:
        """Build sensor group data item"""
        # Build time dataset
        time_dataset = HDF5Dataset(
            name=IMUDataFields.TIME.value,
            data=self.constants["time_data"],
            attributes={},
        )
        # Build imu data dataset
        imu_data_dataset = HDF5Dataset(
            name=IMUDataFields.DATA.value,
            data=self.constants["imu_data"],
            attributes={},
        )
        # Build sensor data group attributes
        sensor_data_group_attr: Dict[str, Any] = {
            IMUDataFields.SENSOR_TYPE.value: sensor_type.value,
            IMUDataFields.ORIENTATION_MAP_SENSOR.value: self.constants[
                "orientation_map_sensor"
            ],
            IMUDataFields.ORIENTATION_MAP_ANATOMICAL.value: self.constants[
                "orientation_map_anatom"
            ],
            IMUDataFields.SAMPLING_RATE.value: self.constants["sampling_rate"],
            IMUDataFields.UNIT.value: self.constants["unit"],
            IMUDataFields.SENSOR_AXIS_NAMES.value: self.constants["axis_names"],
        }
        # Build sensor data group
        return HDF5Group(
            name=sensor_name.value,
            items=[time_dataset, imu_data_dataset],
            attributes=sensor_data_group_attr,
        )

    def create_test_imu_data(self) -> IMUData:
        # Build epoch data
        epoch_data_list: List[EpochIMUData] = [self.__build_epoch_imu_data()]
        # Build metadata
        imu_metadata: IMUMetadata = self.__build_imu_metadata()
        # Add start and end time (inferred from single epoch)
        start_time, end_time = (
            epoch_data_list[0].epoch_start_time,
            epoch_data_list[0].epoch_end_time,
        )
        return IMUData(epoch_data_list, imu_metadata, start_time, end_time)

    def __build_epoch_imu_data(self) -> EpochIMUData:
        sensor_data_list: List[SensorData] = []
        # For every sensor
        for sensor_type, _ in self.constants["sensors"]:
            # Build sensor data
            sensor_data_list.append(self.__build_sensor_data(sensor_type))

        # Get epoch start and end time (inferred from the sensor time axis)
        epoch_start_time, epoch_end_time = (
            sensor_data_list[0].time[0],
            sensor_data_list[0].time[-1],
        )
        return EpochIMUData(sensor_data_list, epoch_start_time, epoch_end_time)

    def __build_sensor_data(self, sensor_type: SensorType) -> SensorData:
        # Get time data.
        time: np.ndarray = np.array(self.constants["time_data"])

        # Build uniaxial sensor data list
        uniaxial_sensor_data_list: List[UniaxialSensorData] = (
            self.__build_uniaxial_sensor_data_list()
        )
        # Build metadata
        sensor_metadata: SensorMetadata = self.__build_sensor_metadata(sensor_type)
        return SensorData(uniaxial_sensor_data_list, time, sensor_metadata)

    def __build_uniaxial_sensor_data_list(self) -> List[UniaxialSensorData]:
        sensor_data: np.ndarray = np.array(self.constants["imu_data"])

        sensor_axis_names: List[SensorCoordinateSystem] = self.constants[
            "model_axis_names"
        ]

        uniaxial_sensor_data_list: List[UniaxialSensorData] = []
        # For axis in sensor axis names
        for index, model_sensor_axis in enumerate(sensor_axis_names):
            uniaxial_sensor_data: np.ndarray = sensor_data[index]
            model_anatomical_axis = self.constants["model_orientation_map"][
                model_sensor_axis
            ]
            uniaxial_sensor_data_list.append(
                UniaxialSensorData(
                    AnatomicalAxis(model_anatomical_axis),
                    SensorAxis(model_sensor_axis),
                    uniaxial_sensor_data,
                )
            )
        return uniaxial_sensor_data_list

    def __build_imu_metadata(self) -> IMUMetadata:
        imu_data_identifier: IMUDataIdentifier = IMUDataIdentifier(
            self.constants["imu_data_id"]
        )
        instrument_name: str = self.constants["instrument_name"]
        instrument_serial_number: str = self.constants["serial_number"]
        model_instrument_id: InstrumentIdentifier = InstrumentIdentifier(
            instrument_name, instrument_serial_number
        )
        return IMUMetadata(imu_data_identifier, model_instrument_id)

    def __build_sensor_metadata(self, sensor_type: SensorType) -> SensorMetadata:
        sampling_rate: float = self.constants["sampling_rate"]
        model_orientation_map: Dict[SensorAxis, AnatomicalAxis] = {
            SensorAxis(sensor_axis): AnatomicalAxis(anatom_axis)
            for sensor_axis, anatom_axis in self.constants[
                "model_orientation_map"
            ].items()
        }
        unit: str = self.constants["unit"]
        return SensorMetadata(sensor_type, sampling_rate, model_orientation_map, unit)


def main():
    """Run test data generation."""
    # Create helper instance
    helper = IMUDataHelper()

    try:
        # Generate test IMU data file
        # file_path = helper.create_test_imu_data_file()
        # print(f"Successfully created test IMU data file at: {file_path}")
        imu_data: IMUData = helper.create_test_imu_data()
        print(imu_data)
    except Exception as e:
        import traceback

        print(f"Error creating test file: {e}")
        print("\nFull traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
