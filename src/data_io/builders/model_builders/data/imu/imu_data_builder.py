import numpy as np
from typing import List, Dict, Tuple

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
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
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier


class IMUDataBuilder(ModelBuilder):
    """Builds IMUData from input HDF5 file"""

    version = "1.0"

    def __init__(self):
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

    def build(self, input_file: HDF5Group, **kwargs) -> IMUData:
        # Assumes single epoch in raw IMU data file
        # Get the sensor data group from the input file
        sensor_data_group: HDF5Group = input_file.get_item_by_name(
            IMUDataFields.SENSOR_DATA.value
        )
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
        # Get time data.
        time: np.ndarray = np.array(
            sensor_data_group.get_item_by_name(IMUDataFields.TIME.value).data
        )
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
        sensor_data: np.ndarray = np.array(
            sensor_data_group.get_item_by_name(IMUDataFields.DATA.value).data
        )
        sensor_axis_names: List[IMUDataFields] = [
            IMUDataFields(axis)
            for axis in sensor_data_group.attributes[IMUDataFields.AXIS_NAMES.value]
        ]
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
        # Convert the file sensor axis to data model sensor axis
        model_sensor_axis: SensorCoordinateSystem = self.file_to_model_sensor_axis_map[
            file_sensor_axis
        ]
        # Get the sensor to anatomical axis map from file
        sensor_to_anatomical_axis_map: Dict[IMUDataFields:IMUDataFields] = (
            sensor_data_group.attributes[IMUDataFields.SENSOR_TO_ANATOMICAL_AXIS_MAP.value]
        )
        # Get the file anatomical axis from the file axis map
        file_anatomical_axis: IMUDataFields = sensor_to_anatomical_axis_map[
            file_sensor_axis
        ]
        # Convert the file anatomical axis to data model anatomical axis
        model_anatomical_axis: AnatomicalCoordinateSystem = (
            self.file_to_model_anatomical_axis_map[file_anatomical_axis]
        )
        return tuple(
            SensorAxis(model_sensor_axis), AnatomicalAxis(model_anatomical_axis)
        )

    def __build_imu_metadata(self, input_file: HDF5Group) -> IMUMetadata:
        imu_data_identifier: IMUDataIdentifier = IMUDataIdentifier(
            input_file.attributes[IMUDataFields.IMU_DATA_IDENTIFIER.value]
        )
        file_instrument_id: Dict[IMUDataFields:str] = input_file.attributes[
            IMUDataFields.INSTRUMENT_IDENTIFIER.value
        ]
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
        file_sensor_orientation_map: Dict[IMUDataFields:IMUDataFields] = {
            IMUDataFields(file_sensor_axis): IMUDataFields(file_anatomical_axis)
            for file_sensor_axis, file_anatomical_axis in sensor_data_group.attributes[
                IMUDataFields.SENSOR_TO_ANATOMICAL_AXIS_MAP
            ].items()
        }
        model_sensor_orientation_map: Dict[SensorAxis:AnatomicalAxis] = {
            SensorAxis(
                self.file_to_model_sensor_axis_map[file_sensor_axis]
            ): AnatomicalAxis(
                self.file_to_model_anatomical_axis_map[file_anatomical_axis]
            )
            for file_sensor_axis, file_anatomical_axis in file_sensor_orientation_map.items()
        }
        unit: str = sensor_data_group.attributes[IMUDataFields.UNIT]
        return SensorMetadata(
            sensor_type, sampling_rate, model_sensor_orientation_map, unit
        )
