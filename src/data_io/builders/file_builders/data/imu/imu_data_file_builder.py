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

    def build(self, data: IMUData) -> HDF5Group:
        if len(data.data) != 0:
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
        # Initialize imu data group
        imu_data_group: HDF5Group = HDF5Group()
        imu_data_group.name = IMUDataFields.IMU_DATA.value
        # Build sensor data group of imu data group
        imu_data_group.items = list(self.__build_sensor_data_group(data.data[0]))
        # Build imu data metadata attributes
        imu_data_group.attributes = self.__build_imu_metadata_attributes(data.metadata)
        return imu_data_group

    def __build_sensor_data_group(self, epoch_imu_data: EpochIMUData) -> HDF5Group:
        """Build the sensor data group containing all sensor measurements.

        Args:
            epoch_imu_data (EpochIMUData): The epoch data containing sensor measurements

        Returns:
            HDF5Group: Group containing all sensor data subgroups
        """
        # Initialize sensor data group
        sensor_data_group: HDF5Group = HDF5Group()
        sensor_data_group.name = IMUDataFields.SENSOR_DATA.value
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
        time.name = IMUDataFields.TIME.value
        time.attributes = {}
        time.data = sensor_data.time.tolist()
        # Build data dataset and add to sensor data subgroup items
        data_dataset: HDF5Dataset = HDF5Dataset()
        data_dataset.name = IMUDataFields.DATA.value
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
            IMUDataFields.IMU_DATA_IDENTIFIER.value: imu_data_identifier,
            IMUDataFields.INSTRUMENT_IDENTIFIER.value: {
                IMUDataFields.INSTRUMENT_NAME.value: instrument_name,
                IMUDataFields.SERIAL_NUMBER.value: serial_number,
            },
        }

    def __build_sensor_metadata_attributes(
        self, sensor_data: SensorData
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
            sensor_type: IMUDataFields = self.model_to_file_sensor_type_map[
                sensor_data.metadata.sensor_type
            ]
        except KeyError:
            raise ValueError(f"Unknown sensor type: {sensor_data.metadata.sensor_type}")

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
            IMUDataFields.SENSOR_TYPE.value: sensor_type,
            IMUDataFields.SENSOR_TO_ANATOMICAL_AXIS_MAP.value: file_axis_map,
            IMUDataFields.SAMPLING_RATE.value: sampling_rate,
            IMUDataFields.UNIT.value: unit,
            IMUDataFields.SENSOR_AXIS_NAMES.value: axis_names,
        }

    def __convert_model_to_file_axis_map(
        self, model_axis_map: Dict[SensorAxis, AnatomicalAxis]
    ) -> Dict[str, str]:
        """Convert model axis mappings to file format mappings.

        Args:
            model_axis_map (Dict[SensorAxis, AnatomicalAxis]): Model's axis mappings

        Returns:
            Dict[str, str]: File format axis mappings

        Raises:
            ValueError: If an unknown axis is encountered
        """
        file_map = {}
        for sensor_axis, anatomical_axis in model_axis_map.items():
            try:
                file_sensor_axis = self.model_to_file_sensor_axis_map[sensor_axis]
                file_anatomical_axis = self.model_to_file_anatomical_axis_map[
                    anatomical_axis
                ]
            except KeyError as e:
                raise ValueError(f"Unknown axis in mapping: {e}")

            file_map[file_sensor_axis] = file_anatomical_axis

        return file_map
