from typing import Dict, List, Tuple

import numpy as np  # type: ignore

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
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


class IMUDataBuilder(ModelBuilder):
    """Builds IMU data model objects from HDF5 file format.

    This builder handles conversion of HDF5 file data into IMU data models,
    including sensor data, metadata, and coordinate system mappings.

    Attributes:
        version (str): Version identifier for the builder
        file_to_model_sensor_axis_map (Dict[IMUDataFields, SensorCoordinateSystem]):
            Maps file fields to sensor coordinate systems
        file_to_model_anatomical_axis_map (Dict[IMUDataFields, AnatomicalCoordinateSystem]):
            Maps file fields to anatomical coordinate systems
        sensor_name_to_sensor_type_map (Dict[IMUDataFields, SensorType]):
            Maps sensor names to types
    """

    version: str = "1.0"

    def __init__(self):
        """Initialize the IMU data builder with mapping dictionaries."""
        super().__init__()
        self.file_to_model_sensor_axis_map: Dict[
            IMUDataFields:SensorCoordinateSystem
        ] = {
            IMUDataFields.SENSOR_AXIS_X: SensorCoordinateSystem.X,
            IMUDataFields.SENSOR_AXIS_Y: SensorCoordinateSystem.Y,
            IMUDataFields.SENSOR_AXIS_Z: SensorCoordinateSystem.Z,
        }
        self.file_to_model_anatomical_axis_map: Dict[
            IMUDataFields:AnatomicalCoordinateSystem
        ] = {
            IMUDataFields.ANATOMICAL_AXIS_ANTEROPOSTERIOR: AnatomicalCoordinateSystem.ANTEROPOSTERIOR,
            IMUDataFields.ANATOMICAL_AXIS_MEDIOLATERAL: AnatomicalCoordinateSystem.MEDIOLATERAL,
            IMUDataFields.ANATOMICAL_AXIS_VERTICAL: AnatomicalCoordinateSystem.VERTICAL,
        }
        self.sensor_name_to_sensor_type_map: Dict[IMUDataFields:SensorType] = {
            IMUDataFields.ACCELEROMETER: SensorType.ACCELEROMETER,
            IMUDataFields.GYROSCOPE: SensorType.GYROSCOPE,
            IMUDataFields.MAGNETOMETER: SensorType.MAGNETOMETER,
        }

    def build(self, input_file: HDF5Group) -> IMUData:
        """Build IMU data model from HDF5 file format.

        Args:
            input_file (HDF5Group): Source HDF5 file data
            **kwargs: Additional build parameters

        Returns:
            IMUData: Constructed IMU data model

        Raises:
            ValueError: If required data is missing or invalid
        """
        if not input_file:
            raise ValueError("Input file data is required")

        # Assumes single epoch in raw IMU data file
        # Get the sensor data group from the input file
        try:
            sensor_data_group: HDF5Group = input_file.get_item_by_name(
                IMUDataFields.SENSOR_DATA.value
            )
        except ValueError as e:
            raise ValueError(f"Missing sensor data group: {e}")

        # Build epoch data
        epoch_data_list: List[EpochIMUData] = [
            self.__build_epoch_imu_data(sensor_data_group)
        ]
        # Build metadata
        imu_metadata: IMUMetadata = self.__build_imu_metadata(input_file)
        # Add start and end time (inferred from single epoch)
        start_time, end_time = (
            epoch_data_list[0].epoch_start_time,
            epoch_data_list[0].epoch_end_time,
        )
        return IMUData(epoch_data_list, imu_metadata, start_time, end_time)

    def __build_epoch_imu_data(self, senor_data_groups: HDF5Group) -> EpochIMUData:
        """Build epoch IMU data from sensor data groups.

        Args:
            senor_data_groups (HDF5Group): Group containing sensor data

        Returns:
            EpochIMUData: Constructed epoch data

        Raises:
            ValueError: If no sensor data is present
        """
        if not senor_data_groups.items:
            raise ValueError("No sensor data found in groups")

        sensor_data_list: List[SensorData] = []
        # For every sensor
        for sensor_data_group in senor_data_groups.items:
            # Build sensor data
            sensor_data_list.append(self.__build_sensor_data(sensor_data_group))

        # Get epoch start and end time (inferred from the sensor time axis)
        epoch_start_time, epoch_end_time = (
            sensor_data_list[0].time[0],
            sensor_data_list[0].time[-1],
        )
        return EpochIMUData(sensor_data_list, epoch_start_time, epoch_end_time)

    def __build_sensor_data(self, sensor_data_group: HDF5Group) -> SensorData:
        """Build sensor data from HDF5 group.

        Args:
            sensor_data_group (HDF5Group): Group containing single sensor data

        Returns:
            SensorData: Constructed sensor data

        Raises:
            ValueError: If required data is missing
        """
        # Get time data.
        try:
            time: np.ndarray = np.array(
                sensor_data_group.get_item_by_name(IMUDataFields.TIME.value).data
            )
        except ValueError as e:
            raise ValueError(f"Missing time data: {e}")

        if len(time) == 0:
            raise ValueError("Empty time data")

        # Build uniaxial sensor data list
        uniaxial_sensor_data_list: List[UniaxialSensorData] = (
            self.__build_uniaxial_sensor_data_list(sensor_data_group)
        )
        # Build metadata
        sensor_metadata: SensorMetadata = self.__build_sensor_metadata(
            sensor_data_group
        )
        return SensorData(uniaxial_sensor_data_list, time, sensor_metadata)

    def __build_uniaxial_sensor_data_list(
        self, sensor_data_group: HDF5Group
    ) -> List[UniaxialSensorData]:
        """Build list of uniaxial sensor data from HDF5 group.

        Args:
            sensor_data_group (HDF5Group): Group containing sensor data

        Returns:
            List[UniaxialSensorData]: List of constructed uniaxial sensor data

        Raises:
            ValueError: If required data or attributes are missing
        """
        try:
            sensor_data: np.ndarray = np.array(
                sensor_data_group.get_item_by_name(IMUDataFields.DATA.value).data
            )
        except ValueError as e:
            raise ValueError(f"Missing sensor data: {e}")

        if IMUDataFields.AXIS_NAMES.value not in sensor_data_group.attributes:
            raise ValueError("Missing axis names in sensor data attributes")

        sensor_axis_names: List[IMUDataFields] = [
            IMUDataFields(axis)
            for axis in sensor_data_group.attributes[IMUDataFields.AXIS_NAMES.value]
        ]

        if not sensor_axis_names:
            raise ValueError("Empty axis names list")

        uniaxial_sensor_data_list: List[UniaxialSensorData] = []
        # For axis in sensor axis names
        for index, file_axis_name in enumerate(sensor_axis_names):
            uniaxial_sensor_data: np.ndarray = sensor_data[index]
            model_sensor_axis, model_anatomical_axis = (
                self.__get_model_axes_from_file_axes(sensor_data_group, file_axis_name)
            )
            uniaxial_sensor_data_list.append(
                UniaxialSensorData(
                    model_anatomical_axis, model_sensor_axis, uniaxial_sensor_data
                )
            )
        return uniaxial_sensor_data_list

    def __get_model_axes_from_file_axes(
        self, sensor_data_group: HDF5Group, file_sensor_axis: IMUDataFields
    ) -> Tuple[SensorAxis, AnatomicalAxis]:
        """Convert file axes to model axes.

        Args:
            sensor_data_group (HDF5Group): Group containing axis mappings
            file_sensor_axis (IMUDataFields): Sensor axis from file

        Returns:
            Tuple[SensorAxis, AnatomicalAxis]: Converted model axes

        Raises:
            ValueError: If axis mapping is invalid or missing
        """
        # Convert the file sensor axis to data model sensor axis
        try:
            model_sensor_axis: SensorCoordinateSystem = (
                self.file_to_model_sensor_axis_map[file_sensor_axis]
            )
        except KeyError:
            raise ValueError(f"Unknown sensor axis: {file_sensor_axis}")

        # Get the sensor to anatomical axis map from file
        if (
            IMUDataFields.SENSOR_TO_ANATOMICAL_AXIS_MAP.value
            not in sensor_data_group.attributes
        ):
            raise ValueError("Missing anatomical axis mapping")

        sensor_to_anatomical_axis_map: Dict[IMUDataFields:IMUDataFields] = (
            sensor_data_group.attributes[
                IMUDataFields.SENSOR_TO_ANATOMICAL_AXIS_MAP.value
            ]
        )
        # Get the file anatomical axis from the file axis map
        try:
            file_anatomical_axis: IMUDataFields = sensor_to_anatomical_axis_map[
                file_sensor_axis
            ]
        except KeyError:
            raise ValueError(
                f"Missing anatomical axis mapping for sensor axis: {file_sensor_axis}"
            )

        # Convert the file anatomical axis to data model anatomical axis
        try:
            model_anatomical_axis: AnatomicalCoordinateSystem = (
                self.file_to_model_anatomical_axis_map[file_anatomical_axis]
            )
        except KeyError:
            raise ValueError(f"Unknown anatomical axis: {file_anatomical_axis}")

        return tuple(
            SensorAxis(model_sensor_axis), AnatomicalAxis(model_anatomical_axis)
        )

    def __build_imu_metadata(self, input_file: HDF5Group) -> IMUMetadata:
        """Build IMU metadata from HDF5 file attributes.

        Args:
            input_file (HDF5Group): Input file containing metadata

        Returns:
            IMUMetadata: Constructed metadata object

        Raises:
            ValueError: If required metadata fields are missing
        """
        if IMUDataFields.IMU_DATA_IDENTIFIER.value not in input_file.attributes:
            raise ValueError("Missing IMU data identifier")

        if IMUDataFields.INSTRUMENT_IDENTIFIER.value not in input_file.attributes:
            raise ValueError("Missing instrument identifier")

        imu_data_identifier: IMUDataIdentifier = IMUDataIdentifier(
            input_file.attributes[IMUDataFields.IMU_DATA_IDENTIFIER.value]
        )
        file_instrument_id: Dict[IMUDataFields:str] = input_file.attributes[
            IMUDataFields.INSTRUMENT_IDENTIFIER.value
        ]

        if IMUDataFields.INSTRUMENT_NAME not in file_instrument_id:
            raise ValueError("Missing instrument name")
        if IMUDataFields.SERIAL_NUMBER not in file_instrument_id:
            raise ValueError("Missing serial number")

        instrument_name: str = file_instrument_id[IMUDataFields.INSTRUMENT_NAME]
        instrument_serial_number: str = file_instrument_id[IMUDataFields.SERIAL_NUMBER]
        model_instrument_id: InstrumentIdentifier = InstrumentIdentifier(
            instrument_name, instrument_serial_number
        )
        return IMUMetadata(imu_data_identifier, model_instrument_id)

    def __build_sensor_metadata(self, sensor_data_group: HDF5Group) -> SensorMetadata:
        sensor_type: SensorType = self.sensor_name_to_sensor_type_map[
            IMUDataFields(sensor_data_group.name)
        ]
        sampling_rate: float = sensor_data_group.attributes[IMUDataFields.SAMPLING_RATE]
        # Construct file sensor orientation map from input file lists
        file_orientation_map: Dict[IMUDataFields, IMUDataFields] = {
            IMUDataFields(sensor_axis): IMUDataFields(anatomical_axis)
            for sensor_axis, anatomical_axis in zip(
                sensor_data_group.attributes[
                    IMUDataFields.ORIENTATION_MAP_SENSOR.value
                ],
                sensor_data_group.attributes[
                    IMUDataFields.ORIENTATION_MAP_ANATOMICAL.value
                ],
            )
        }
        # Convert the file map to model map
        model_orientation_map: Dict[SensorAxis, AnatomicalAxis] = {
            SensorAxis(
                self.file_to_model_sensor_axis_map(file_sensor_axis)
            ): AnatomicalAxis(self.file_to_model_anatomical_axis_map(file_anatom_axis))
            for file_sensor_axis, file_anatom_axis in file_orientation_map
        }
        unit: str = sensor_data_group.attributes[IMUDataFields.UNIT]
        return SensorMetadata(sensor_type, sampling_rate, model_orientation_map, unit)
