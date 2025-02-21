from typing import Any, Dict, List

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_types.instrument.sensor_type import SensorType
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


class IMUDataFileBuilder(FileBuilder):
    """Builds HDF5 file format from IMU data model objects.

    This builder handles conversion of IMU (Inertial Measurement Unit) data
    into HDF5 file format, including sensor data and metadata.

    Attributes:
        version (str): Version identifier for the builder
        model_to_file_sensor_type_map (Dict[SensorType, str]): Maps sensor types to file fields
        model_to_file_anatomical_axis_map (Dict[AnatomicalCoordinateSystem, str]): Maps anatomical axes
        model_to_file_sensor_axis_map (Dict[SensorCoordinateSystem, str]): Maps sensor axes
    """

    version: str = "1.0"

    def __init__(self) -> None:
        """Initialize the IMU data file builder with mapping dictionaries.

        Raises:
            ValueError: If mapping dictionaries are incomplete
        """
        super().__init__()
        self.model_to_file_sensor_type_map: Dict[SensorType, str] = {
            SensorType.ACCELEROMETER: IMUDataFields.ACCELEROMETER.value,
            SensorType.GYROSCOPE: IMUDataFields.GYROSCOPE.value,
            SensorType.MAGNETOMETER: IMUDataFields.MAGNETOMETER.value,
        }
        self.model_to_file_anatomical_axis_map: Dict[
            AnatomicalCoordinateSystem, str
        ] = {
            AnatomicalCoordinateSystem.ANTEROPOSTERIOR: IMUDataFields.ANATOMICAL_AXIS_ANTEROPOSTERIOR.value,
            AnatomicalCoordinateSystem.MEDIOLATERAL: IMUDataFields.ANATOMICAL_AXIS_MEDIOLATERAL.value,
            AnatomicalCoordinateSystem.VERTICAL: IMUDataFields.ANATOMICAL_AXIS_VERTICAL.value,
        }
        self.model_to_file_sensor_axis_map: Dict[SensorCoordinateSystem, str] = {
            SensorCoordinateSystem.X: IMUDataFields.SENSOR_AXIS_X.value,
            SensorCoordinateSystem.Y: IMUDataFields.SENSOR_AXIS_Y.value,
            SensorCoordinateSystem.Z: IMUDataFields.SENSOR_AXIS_Z.value,
        }

        # Validate that all sensor types are mapped
        if not all(
            sensor_type in self.model_to_file_sensor_type_map
            for sensor_type in SensorType
        ):
            raise ValueError("Incomplete sensor type mapping")

        # Validate that all axes are mapped
        if not all(
            axis in self.model_to_file_anatomical_axis_map
            for axis in AnatomicalCoordinateSystem
        ):
            raise ValueError("Incomplete anatomical axis mapping")

        if not all(
            axis in self.model_to_file_sensor_axis_map
            for axis in SensorCoordinateSystem
        ):
            raise ValueError("Incomplete sensor axis mapping")

    def build(self, data: IMUData) -> HDF5Group:
        if not isinstance(data, IMUData) or len(data.data) != 1:
            raise ValueError("File must contain single epoch")
        return self.__build_imu_data_group(data)

    def __build_imu_data_group(self, data: IMUData) -> HDF5Group:
        """Build the main IMU data group in HDF5 format.

        Args:
            data (IMUData): The IMU data model to convert

        Returns:
            HDF5Group: The root group containing all IMU data

        Raises:
            ValueError: If data validation fails
        """
        # Get data group name
        imu_data_group_name = IMUDataFields.IMU_DATA.value
        # Build sensor data group of imu data group
        imu_data_group_items = [self.__build_sensor_data_group(data.data[0])]
        # Build imu data metadata attributes
        imu_data_group_attributes = self.__build_imu_metadata_attributes(data.metadata)
        return HDF5Group(
            name=imu_data_group_name,
            items=imu_data_group_items,
            attributes=imu_data_group_attributes,
        )

    def __build_sensor_data_group(self, epoch_imu_data: EpochIMUData) -> HDF5Group:
        """Build the sensor data group containing all sensor measurements.

        Args:
            epoch_imu_data (EpochIMUData): The epoch data containing sensor measurements

        Returns:
            HDF5Group: Group containing all sensor data subgroups
        """
        # Initialize sensor data group
        sensor_data_group_name = IMUDataFields.SENSOR_DATA.value
        sensor_data_group_items = []
        sensor_data_group_attributes = {}

        # For every sensor present in IMU data
        for sensor_data in epoch_imu_data.data:
            # Build sensor subgroup and append to sensor data group items
            sensor_data_group_items.append(
                self.__build_sensor_data_subgroup(sensor_data)
            )

        return HDF5Group(
            name=sensor_data_group_name,
            items=sensor_data_group_items,
            attributes=sensor_data_group_attributes,
        )

    def __build_sensor_data_subgroup(self, sensor_data: SensorData) -> HDF5Group:
        # Initialize sensor subgroup
        sensor_data_subgroup_name = self.model_to_file_sensor_type_map[
            sensor_data.metadata.sensor_type
        ]
        # Build time dataset and add to sensor data subgroup items
        time: HDF5Dataset = HDF5Dataset(
            name=IMUDataFields.TIME.value, data=sensor_data.time.tolist(), attributes={}
        )
        # Build data dataset and add to sensor data subgroup items
        sensor_axis_names: List[SensorAxis] = []
        axis_data: List[List[Any]] = []
        for axis in sensor_data.data:
            sensor_axis_names.append(axis.sensor_axis)
            axis_data.append(axis.data.tolist())
        data_dataset: HDF5Dataset = HDF5Dataset(
            name=IMUDataFields.DATA.value,
            data=axis_data,
            attributes={},
        )
        # Add time and data datasets to sensor data subgroup's items
        sensor_data_subgroup_items = [time, data_dataset]
        # Build sensor metadata attributes
        sensor_data_subgroup_attributes = self.__build_sensor_metadata_attributes(
            sensor_data, sensor_axis_names
        )
        return HDF5Group(
            name=sensor_data_subgroup_name,
            items=sensor_data_subgroup_items,
            attributes=sensor_data_subgroup_attributes,
        )

    def __build_imu_metadata_attributes(
        self, imu_metadata: IMUMetadata
    ) -> Dict[str, Any]:
        imu_data_identifier: str = imu_metadata.imu_data_identifier.value
        instrument_name = imu_metadata.instrument_identifier.name
        serial_number = imu_metadata.instrument_identifier.serial_number
        return {
            IMUDataFields.IMU_DATA_IDENTIFIER.value: imu_data_identifier,
            IMUDataFields.INSTRUMENT_NAME.value: instrument_name,
            IMUDataFields.SERIAL_NUMBER.value: serial_number,
        }

    def __build_sensor_metadata_attributes(
        self, sensor_data: SensorData, sensor_axis_names: List[SensorAxis]
    ) -> Dict[str, Any]:
        """Build metadata attributes for a sensor.

        Args:
            sensor_data (SensorData): The sensor data containing metadata

        Returns:
            Dict[str, Any]: Dictionary of metadata attributes

        Raises:
            ValueError: If required metadata fields are missing
        """
        if not sensor_data.metadata:
            raise ValueError("Sensor metadata is required")

        if not sensor_data.metadata.sensor_type:
            raise ValueError("Sensor type is required in metadata")

        # Get file sensor type
        try:
            sensor_type: str = self.model_to_file_sensor_type_map[
                sensor_data.metadata.sensor_type
            ]
        except KeyError:
            raise ValueError(f"Unknown sensor type: {sensor_data.metadata.sensor_type}")

        # Construct orientation map entries
        orientation_map_sensor: List[str] = []
        orientation_map_anatomical: List[str] = []
        sensor_axes = [
            (data.sensor_axis.name, data.anatomical_axis.name)
            for data in sensor_data.data
        ]
        for sensor_axis, anatom_axis in sensor_axes:
            orientation_map_sensor.append(
                self.model_to_file_sensor_axis_map[sensor_axis]
            )
            orientation_map_anatomical.append(
                self.model_to_file_anatomical_axis_map[anatom_axis]
            )

        # Get sampling rate
        sampling_rate: float = sensor_data.metadata.sampling_rate
        # Get unit
        unit: str = sensor_data.metadata.unit
        # Conver model sensor axis names to file sensor axis names
        sensor_axis_names: List[str] = [
            self.model_to_file_sensor_axis_map[axis.name] for axis in sensor_axis_names
        ]
        return {
            IMUDataFields.SENSOR_TYPE.value: sensor_type,
            IMUDataFields.ORIENTATION_MAP_SENSOR.value: orientation_map_sensor,
            IMUDataFields.ORIENTATION_MAP_ANATOMICAL.value: orientation_map_anatomical,
            IMUDataFields.SAMPLING_RATE.value: sampling_rate,
            IMUDataFields.UNIT.value: unit,
            IMUDataFields.AXIS_NAMES.value: sensor_axis_names,
        }
