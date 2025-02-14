from typing import Any, Dict, List, Union

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
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


class IMUDataFileBuilder(FileBuilder):
    version = "1.0"

    def __init__(self):
        self.model_to_file_sensor_type_map: Dict[SensorType, IMUDataFields] = {
            SensorType.ACCELEROMETER: IMUDataFields.ACCELEROMETER,
            SensorType.GYROSCOPE: IMUDataFields.GYROSCOPE,
            SensorType.MAGNETOMETER: IMUDataFields.MAGNETOMETER,
        }
        self.model_to_file_anatomical_axis_map: Dict[
            AnatomicalCoordinateSystem:IMUDataFields
        ] = {
            AnatomicalCoordinateSystem.ANTEROPOSTERIOR: IMUDataFields.ANATOMICAL_AXIS_ANTEROPOSTERIOR,
            AnatomicalCoordinateSystem.MEDIOLATERAL: IMUDataFields.ANATOMICAL_AXIS_MEDIOLATERAL,
            AnatomicalCoordinateSystem.VERTICAL: IMUDataFields.ANATOMICAL_AXIS_VERTICAL,
        }
        self.model_to_file_sensor_axis_map: Dict[
            SensorCoordinateSystem:IMUDataFields
        ] = {
            SensorCoordinateSystem.X: IMUDataFields.SENSOR_AXIS_X,
            SensorCoordinateSystem.Y: IMUDataFields.SENSOR_AXIS_Y,
            SensorCoordinateSystem.Z: IMUDataFields.SENSOR_AXIS_Z,
        }

    def build(self, data: IMUData) -> HDF5Group:
        if len(IMUData.data) != 0:
            raise ValueError("File must contain single epoch")
        return self.__build_imu_data_group(data)

    def __build_imu_data_group(self, data: IMUData) -> HDF5Group:
        # Initialize imu data group
        imu_data_group: HDF5Group = HDF5Group
        imu_data_group.name = IMUDataFields.IMU_DATA
        # Build sensor data group of imu data group
        imu_data_group.items = list(self.__build_sensor_data_group(IMUData.data[0]))
        # Build imu data metadata attributes
        imu_data_group.attributes = self.__build_imu_metadata_attributes(
            IMUData.metadata
        )
        return imu_data_group

    def __build_sensor_data_group(self, epoch_imu_data: EpochIMUData) -> HDF5Group:
        # Initialize sensor data group
        sensor_data_group: HDF5Group = HDF5Group()
        sensor_data_group.name = IMUDataFields.SENSOR_DATA
        sensor_data_group.items = []
        sensor_data_group.attributes = {}

        # For every sensor present in IMU data
        for sensor_data in epoch_imu_data.data:
            # Build sensor subgroup and append to sensor data group items
            sensor_data_group.items.append(
                self.__build_sensor_data_subgroup(sensor_data)
            )

        return sensor_data_group

    def __build_sensor_data_subgroup(self, sensor_data: SensorData) -> HDF5Group:
        # Initialize sensor subgroup
        sensor_data_subgroup: HDF5Group = HDF5Group()
        sensor_data_subgroup.name = self.model_to_file_sensor_type_map[
            SensorData.metadata.sensor_type
        ]
        # Build time dataset and add to sensor data subgroup items
        time: HDF5Dataset = HDF5Dataset()
        time.name = IMUDataFields.TIME
        time.attributes = {}
        time.data = sensor_data.time.tolist()
        # Build data dataset and add to sensor data subgroup items
        data_dataset: HDF5Dataset = HDF5Dataset()
        data_dataset.name = IMUDataFields.DATA
        data_dataset.attributes = {}
        data_dataset.data = [axis.data.tolist() for axis in sensor_data.data]
        # Add time and data datasets to sensor data subgroup's items
        sensor_data_subgroup.items = [time, data_dataset]
        # Build sensor metadata attributes
        sensor_data_subgroup.attributes = self.__build_sensor_metadata_attributes(
            sensor_data
        )
        return sensor_data_subgroup

    def __build_imu_metadata_attributes(
        self, imu_metadata: IMUMetadata
    ) -> Dict[str, Any]:
        imu_data_identifier: str = IMUMetadata.imu_data_identifier.value
        instrument_id: str = IMUMetadata.instument_identifier.value
        instrument_name = instrument_id.split("_")[0]
        serial_number = instrument_id.split("_")[1]
        return {
            IMUDataFields.IMU_DATA_IDENTIFIER: imu_data_identifier,
            IMUDataFields.INSTRUMENT_IDENTIFIER: {
                IMUDataFields.INSTRUMENT_NAME: instrument_name,
                IMUDataFields.SERIAL_NUMBER: serial_number,
            },
        }

    def __build_sensor_metadata_attributes(
        self, sensor_data: SensorData
    ) -> Dict[str, Any]:
        # Get file senor type
        sensor_type: IMUDataFields = self.model_to_file_sensor_type_map[
            sensor_data.metadata.sensor_type
        ]
        # Get file axis map
        file_axis_map: Dict[IMUDataFields, IMUDataFields] = (
            self.__convert_model_to_file_axis_map(
                sensor_data.metadata.sensor_orientation_map
            )
        )
        # Get sampling rate
        sampling_rate: float = sensor_data.metadata.sampling_rate
        # Get unit
        unit: str = sensor_data.metadata.unit
        # Get file axis names
        axis_names: List[IMUDataFields] = [
            self.model_to_file_sensor_type_map[axis.sensor_axis]
            for axis in sensor_data.data
        ]
        return {
            IMUDataFields.SENSOR_TYPE: sensor_type,
            IMUDataFields.SENSOR_TO_ANATOMICAL_AXIS_MAP: file_axis_map,
            IMUDataFields.SAMPLING_RATE: sampling_rate,
            IMUDataFields.UNIT: unit,
            IMUDataFields.SENSOR_AXIS_NAMES: axis_names,
        }

    def __convert_model_to_file_axis_map(
        self, model_axis_map: Dict[SensorAxis, AnatomicalAxis]
    ) -> Dict[IMUDataFields, IMUDataFields]:
        return {
            self.model_to_file_sensor_axis_map[
                sensor_axis
            ]: self.model_to_file_anatomical_axis_map[anatomical_axis]
            for sensor_axis, anatomical_axis in model_axis_map
        }
