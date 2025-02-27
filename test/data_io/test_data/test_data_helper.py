import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_io.model_fields.data.user.clinical_demographic_data_fields import (
    ClinicalDemographicDataFields,
)
from src.data_io.model_fields.data.user.user_data_fields import UserDataFields
from src.data_io.model_fields.dataset.dataset_fields import DatasetFields
from src.data_io.model_fields.feature_set.feature_set_fields import FeatureSetFields
from src.data_io.model_fields.features.aggregate.aggregate_feature_fields import (
    AggregateFeatureFields,
)
from src.data_io.model_fields.features.raw.raw_feature_fields import RawFeatureFields
from src.data_io.model_fields.instrument_specification.instrument_specification_fields import (
    InstrumentSpecificationFields,
)
from src.data_io.model_fields.mappings.mapping_fields import MappingFields
from src.data_io.model_fields.registries.registry_fields import RegistryFields
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_model.data.user.clinical.clinical_assessment import ClinicalAssessment
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
    Sex,
)
from src.data_model.data.user.user_data import UserData
from src.data_model.dataset.dataset import Dataset
from src.data_model.dataset.dataset_entry import DatasetEntry
from src.data_model.feature_set.feature_set import FeatureSet
from src.data_model.feature_set.feature_set_entry import FeatureSetEntry
from src.data_model.features.aggregate.aggregate_feature import AggregateFeature
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.aggregate.descriptive_statistic import DescriptiveStatistic
from src.data_model.features.aggregate.metadata.aggregate_feature_set_entry_metadata import (
    AggregateFeatureSetEntryMetadata,
)
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_epoch_features import RawEpochFeatures
from src.data_model.features.raw.raw_feature import RawFeature
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_model.instrument_specifications.imu_specifications import (
    IMUSpecifications,
)
from src.data_model.instrument_specifications.sensor_specifications import (
    SensorSpecification,
)
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier
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
from src.util.mechanics.units.si.kilogram import Kilogram
from src.util.mechanics.units.si.meter import Meter


class TestConstants(Enum):
    """Test constants for IMU data generation."""

    ############### IMU DATA ###############
    IMU_DATA_ID = "test_imu_data_identifier"
    INSTRUMENT_NAME = "testInstrumentName"
    SERIAL_NUMBER = "1234567890"
    SAMPLING_RATE = 100.0
    UNIT = "test_unit"
    TIME_DATA = [float(i) / 10 for i in range(0, 11, 1)]
    IMU_DATA = [
        [0.0 for i in range(10)],
        [1.0 for i in range(10)],
        [2.0 for i in range(10)],
    ]

    # Sensor types and names
    SENSORS = [
        (SensorType.ACCELEROMETER, IMUDataFields.ACCELEROMETER),
        (SensorType.GYROSCOPE, IMUDataFields.GYROSCOPE),
    ]

    # Axis mappings
    ORIENTATION_MAP_SENSOR = [
        IMUDataFields.SENSOR_AXIS_X.value,
        IMUDataFields.SENSOR_AXIS_Y.value,
        IMUDataFields.SENSOR_AXIS_Z.value,
    ]
    ORIENTATION_MAP_ANATOM = [
        IMUDataFields.ANATOMICAL_AXIS_ANTEROPOSTERIOR.value,
        IMUDataFields.ANATOMICAL_AXIS_MEDIOLATERAL.value,
        IMUDataFields.ANATOMICAL_AXIS_VERTICAL.value,
    ]
    AXIS_NAMES = [
        IMUDataFields.SENSOR_AXIS_X.value,
        IMUDataFields.SENSOR_AXIS_Y.value,
        IMUDataFields.SENSOR_AXIS_Z.value,
    ]
    MODEL_AXIS_NAMES = [
        SensorCoordinateSystem.X,
        SensorCoordinateSystem.Y,
        SensorCoordinateSystem.Z,
    ]
    MODEL_ORIENTATION_MAP = {
        SensorCoordinateSystem.X: AnatomicalCoordinateSystem.ANTEROPOSTERIOR,
        SensorCoordinateSystem.Y: AnatomicalCoordinateSystem.MEDIOLATERAL,
        SensorCoordinateSystem.Z: AnatomicalCoordinateSystem.VERTICAL,
    }
    IMU_FILE_NAME = "test_imu_data.h5"
    AGGREGATE_FILE_NAME = "test_aggregate_feature.h5"
    RAW_FILE_NAME = "test_raw_feature.h5"

    ############### Feature DATA ###############
    RAW_FEATURE_NAMES = [
        RawFeatureType.PLACEHOLDER.value,
        RawFeatureType.PLACEHOLDER.value,
        RawFeatureType.PLACEHOLDER.value,
    ]
    EPOCH_START_TIMES = [0.0, 1.0, 2.0]
    STAT_NAMES = [
        DescriptiveStatisticType.PLACEHOLDER.value,
        DescriptiveStatisticType.PLACEHOLDER.value,
        DescriptiveStatisticType.PLACEHOLDER.value,
    ]
    FEATURE_DATA = [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    FEATURE_IMU_DATA_ID = "test_feature_imu_data_id"
    FEATURE_USER_DATA_ID = "test_user_data_id"
    RAW_FEATURE_ID = "test_raw_feature_id"
    AGG_FEATURE_ID = "test_agg_feature_id"
    RAW_FEATURE_START_TIME = 0.0
    RAW_FEATURE_EPOCH_LEN = 1.0
    PLACEHOLDER_STAT_VALUE = 1.0
    PLACEHOLDER_FEATURE_VALUE = 2.0

    ############### User DATA ###############
    USER_DATA_ID = "test_user_data_id"
    USER_NAME = "Test User"
    USER_AGE = 65.0
    USER_SEX = "male"  # lowercase to match Sex enum values
    USER_WEIGHT = 70.0  # kg
    USER_HEIGHT = 1.75  # m

    ############### Dataset ###############
    DATASET_NAME = "test_dataset"
    DATASET_USER_IDS = ["user_1", "user_2", "user_3"]
    DATASET_IMU_IDS = ["imu_1", "imu_2", "imu_3"]

    ############### Feature Set ###############
    FEATURE_SET_NAME = "test_feature_set"
    FEATURE_SET_RAW_IDS = ["raw_1", "raw_2", "raw_3"]
    FEATURE_SET_AGG_IDS = ["agg_1", "agg_2", "agg_3"]

    ############### Instrument Specifications ###############
    SPEC_ID = "test_spec_id"
    IMU_NAME = "Test IMU"
    SENSOR_NAME = "Test Sensor"
    SENSOR_UNITS = "m/s^2"
    SENSOR_RANGE = (-16.0, 16.0)
    SENSOR_SENSITIVITY = 2048.0  # LSB/unit
    SENSOR_RESOLUTION = 16  # bits
    SENSOR_SAMPLING_RATE = 100.0  # Hz
    SENSOR_NOISE_DENSITY = 0.0004  # unit/√Hz
    SENSOR_BIAS_STABILITY = 0.1  # unit
    SENSOR_ALIGNMENT_ERROR = 0.1  # degrees
    SENSOR_CROSS_AXIS_SENSITIVITY = 2.0  # %
    SENSOR_POWER_CONSUMPTION = 0.45  # mW
    SENSOR_OPERATING_CONDITIONS = {
        "temperature": "-40°C to 85°C",
        "humidity": "10% to 90%",
    }
    SENSOR_SIZE = {"length": 3.0, "width": 3.0, "height": 0.95}
    SENSOR_MASS = 0.3  # grams

    ############### Mappings ###############
    # IMU to User mapping
    IMU_TO_USER_SOURCE_IDS = ["imu_1", "imu_2", "imu_3"]
    IMU_TO_USER_TARGET_IDS = ["user_1", "user_2", "user_3"]

    # Raw to IMU mapping
    RAW_TO_IMU_SOURCE_IDS = ["raw_1", "raw_2", "raw_3"]
    RAW_TO_IMU_TARGET_IDS = ["imu_1", "imu_2", "imu_3"]

    # Aggregate to Raw mapping
    AGG_TO_RAW_SOURCE_IDS = ["agg_1", "agg_2", "agg_3"]
    AGG_TO_RAW_TARGET_IDS = ["raw_1", "raw_2", "raw_3"]

    # Instrument to Spec mapping
    INST_TO_SPEC_SOURCE_IDS = ["manufacturer1_serial1", "manufacturer2_serial2"]
    INST_TO_SPEC_TARGET_IDS = ["spec_1", "spec_2"]

    ############### Registries ###############
    # Common registry paths
    REGISTRY_PATHS = ["path/to/file1", "path/to/file2", "path/to/file3"]

    # Registry IDs
    USER_REGISTRY_IDS = ["user_1", "user_2", "user_3"]
    IMU_REGISTRY_IDS = ["imu_1", "imu_2", "imu_3"]
    RAW_FEATURE_REGISTRY_IDS = ["raw_1", "raw_2", "raw_3"]
    AGG_FEATURE_REGISTRY_IDS = ["agg_1", "agg_2", "agg_3"]
    INSTRUMENT_SPEC_REGISTRY_IDS = ["spec_1", "spec_2", "spec_3"]

    ############### File I/O ###############
    # CSV Test Data
    CSV_FIELDNAMES = ["column1", "column2", "column3"]
    CSV_DATA = {
        "column1": ["value1", "value2", "value3"],
        "column2": ["value4", "value5", "value6"],
        "column3": ["value7", "value8", "value9"],
    }

    # JSON Test Data
    JSON_DATA = {
        "string_field": "test_string",
        "number_field": 42,
        "list_field": [1, 2, 3],
        "nested_field": {"inner_field": "inner_value"},
    }

    # HDF5 Test Data
    HDF5_DATASET_1_NAME = "test_dataset_1"
    HDF5_DATASET_2_NAME = "test_dataset_2"
    HDF5_DATA = [[1, 2, 3], [4, 5, 6]]
    HDF5_ATTRIBUTES = {"attr1": "value1", "attr2": 42}

    HDF5_GROUP_NAME = "test_group_name"
    HDF5_GROUP_ATTRIBUTES = {"attr1": "value1", "attr2": 99}


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

    def create_test_imu_data_file(self, path: Optional[Path] = None) -> Path:
        """Create test IMU data file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        # Create test data
        test_data = self.create_test_imu_data_hdf5()

        # Use provided path or default
        if path is None:
            path = Path("test/test_data/imu_data.h5")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        writer = HDF5FileWriter()
        success, error = writer.write(path, test_data)
        if not success:
            raise RuntimeError(f"Failed to write test IMU data: {error}")

        return path

    def create_test_imu_data_hdf5(self) -> HDF5Group:
        """Creates test imu data in HDF5Group"""
        return HDF5Group(
            name=IMUDataFields.IMU_DATA.value,
            attributes=self._build_test_imu_data_attributes(),
            items=self._build_sensor_data_group(),
        )

    def _build_test_imu_data_attributes(self) -> Dict[str, Any]:
        """Build test IMU data attributes."""
        test_attributes = {}
        test_attributes[IMUDataFields.IMU_DATA_IDENTIFIER.value] = (
            TestConstants.IMU_DATA_ID.value
        )
        test_attributes[IMUDataFields.INSTRUMENT_NAME.value] = (
            TestConstants.INSTRUMENT_NAME.value
        )
        test_attributes[IMUDataFields.SERIAL_NUMBER.value] = (
            TestConstants.SERIAL_NUMBER.value
        )
        return test_attributes

    def _build_sensor_data_group(self) -> List[HDF5Group]:
        """Build test IMU data items."""
        sensor_data_group = HDF5Group(
            name=IMUDataFields.SENSOR_DATA.value, items=[], attributes={}
        )
        items = []
        for sensor_type, sensor_name in TestConstants.SENSORS.value:
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
            data=TestConstants.TIME_DATA.value,
            attributes={},
        )
        # Build imu data dataset
        imu_data_dataset = HDF5Dataset(
            name=IMUDataFields.DATA.value,
            data=TestConstants.IMU_DATA.value,
            attributes={},
        )
        # Build sensor data group attributes
        sensor_data_group_attr: Dict[str, Any] = {
            IMUDataFields.SENSOR_TYPE.value: sensor_type.value,
            IMUDataFields.ORIENTATION_MAP_SENSOR.value: TestConstants.ORIENTATION_MAP_SENSOR.value,
            IMUDataFields.ORIENTATION_MAP_ANATOMICAL.value: TestConstants.ORIENTATION_MAP_ANATOM.value,
            IMUDataFields.SAMPLING_RATE.value: TestConstants.SAMPLING_RATE.value,
            IMUDataFields.UNIT.value: TestConstants.UNIT.value,
            IMUDataFields.AXIS_NAMES.value: TestConstants.AXIS_NAMES.value,
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
        for sensor_type, _ in TestConstants.SENSORS.value:
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
        time: np.ndarray = np.array(TestConstants.TIME_DATA.value)

        # Build uniaxial sensor data list
        uniaxial_sensor_data_list: List[UniaxialSensorData] = (
            self.__build_uniaxial_sensor_data_list()
        )
        # Build metadata
        sensor_metadata: SensorMetadata = self.__build_sensor_metadata(sensor_type)
        return SensorData(uniaxial_sensor_data_list, time, sensor_metadata)

    def __build_uniaxial_sensor_data_list(self) -> List[UniaxialSensorData]:
        sensor_data: np.ndarray = np.array(TestConstants.IMU_DATA.value)

        sensor_axis_names: List[SensorCoordinateSystem] = (
            TestConstants.MODEL_AXIS_NAMES.value
        )

        uniaxial_sensor_data_list: List[UniaxialSensorData] = []
        # For axis in sensor axis names
        for index, model_sensor_axis in enumerate(sensor_axis_names):
            uniaxial_sensor_data: np.ndarray = sensor_data[index]
            model_anatomical_axis = TestConstants.MODEL_ORIENTATION_MAP.value[
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
            TestConstants.IMU_DATA_ID.value
        )
        instrument_name: str = TestConstants.INSTRUMENT_NAME.value
        instrument_serial_number: str = TestConstants.SERIAL_NUMBER.value
        model_instrument_id: InstrumentIdentifier = InstrumentIdentifier(
            instrument_name, instrument_serial_number
        )
        return IMUMetadata(imu_data_identifier, model_instrument_id)

    def __build_sensor_metadata(self, sensor_type: SensorType) -> SensorMetadata:
        sampling_rate: float = TestConstants.SAMPLING_RATE.value
        unit: str = TestConstants.UNIT.value
        return SensorMetadata(sensor_type, sampling_rate, unit)


class FeatureDataHelper:
    """Helper class for creating test feature data."""

    def __init__(self):
        self.test_data_helper = TestDataHelper()

    ############### AGGREGATE ###############
    def create_test_aggregate_feature_file(self, path: Optional[Path] = None) -> Path:
        """Create test aggregate feature file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/aggregate_feature.h5")

        path.parent.mkdir(parents=True, exist_ok=True)

        writer = HDF5FileWriter()
        test_data = self.create_test_aggregate_feature_hdf5()
        success, error = writer.write(path, test_data)
        if not success:
            raise RuntimeError(f"Failed to write test aggregate feature file: {error}")

        return path

    def create_test_aggregate_feature_hdf5(self) -> HDF5Group:
        # Build feature dataset
        feature_dataset: HDF5Dataset = self.__build_agg_feature_dataset()
        # Build feature names (rows, feature names)
        feature_names: HDF5Dataset = self.__build_agg_feature_names()
        # Build stat names (cols, stat names)
        stat_names: HDF5Dataset = self.__build_stat_names()
        # Build attributes
        attributes: Dict[str, str] = self.__build_agg_attributes()
        return HDF5Group(
            name=AggregateFeatureFields.AGGREGATE_FEATURE.value,
            items=[feature_dataset, feature_names, stat_names],
            attributes=attributes,
        )

    def __build_agg_feature_dataset(self) -> HDF5Dataset:
        return HDF5Dataset(
            name=AggregateFeatureFields.FEATURES.value,
            data=TestConstants.FEATURE_DATA.value,
            attributes={},
        )

    def __build_agg_feature_names(self) -> HDF5Dataset:
        return HDF5Dataset(
            name=AggregateFeatureFields.FEATURE_NAMES.value,
            data=TestConstants.RAW_FEATURE_NAMES.value,
            attributes={},
        )

    def __build_stat_names(self) -> HDF5Dataset:
        return HDF5Dataset(
            name=AggregateFeatureFields.DESCRIPTIVE_STATISTIC_NAMES.value,
            data=TestConstants.STAT_NAMES.value,
            attributes={},
        )

    def __build_agg_attributes(self) -> Dict[str, str]:
        return {
            AggregateFeatureFields.AGGREGATE_FEATURE_IDENTIFIER.value: TestConstants.AGG_FEATURE_ID.value,
            AggregateFeatureFields.RAW_FEATURE_IDENTIFIER.value: TestConstants.RAW_FEATURE_ID.value,
            AggregateFeatureFields.IMU_DATA_IDENTIFIER.value: TestConstants.FEATURE_IMU_DATA_ID.value,
            AggregateFeatureFields.USER_DATA_IDENTIFIER.value: TestConstants.FEATURE_USER_DATA_ID.value,
        }

    def create_test_aggregate_feature(self) -> AggregateFeatureSetEntry:
        return AggregateFeatureSetEntry(
            aggregate_features=self.__build_aggregate_feature_list(),
            metadata=self.__build_aggregate_metadata(),
        )

    def __build_aggregate_feature_list(self) -> List[AggregateFeature]:
        return [
            AggregateFeature(
                descriptive_statistics=self.__build_descriptive_statistics_list(),
                feature_type=RawFeatureType.PLACEHOLDER,
            ),
            AggregateFeature(
                descriptive_statistics=self.__build_descriptive_statistics_list(),
                feature_type=RawFeatureType.PLACEHOLDER,
            ),
        ]

    def __build_descriptive_statistics_list(self) -> List[DescriptiveStatistic]:
        return [
            DescriptiveStatistic(
                statistic_type=DescriptiveStatisticType.PLACEHOLDER,
                value=TestConstants.PLACEHOLDER_STAT_VALUE.value,
            ),
            DescriptiveStatistic(
                statistic_type=DescriptiveStatisticType.PLACEHOLDER,
                value=TestConstants.PLACEHOLDER_STAT_VALUE.value + 1.0,
            ),
        ]

    def __build_aggregate_metadata(self) -> AggregateFeatureSetEntryMetadata:
        return AggregateFeatureSetEntryMetadata(
            aggregate_feature_identifier=AggregateFeatureIdentifier(
                TestConstants.AGG_FEATURE_ID.value
            ),
            raw_feature_identifier=RawFeatureIdentifier(
                TestConstants.RAW_FEATURE_ID.value
            ),
            user_identifier=UserIdentifier(TestConstants.FEATURE_USER_DATA_ID.value),
            imu_data_identifier=IMUDataIdentifier(
                TestConstants.FEATURE_IMU_DATA_ID.value
            ),
        )

    ############### RAW ###############
    def create_test_raw_feature_file(self, path: Optional[Path] = None) -> Path:
        """Create test raw feature file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/raw_feature.h5")

        path.parent.mkdir(parents=True, exist_ok=True)

        writer = HDF5FileWriter()
        test_data = self.create_test_raw_feature_hdf5()
        success, error = writer.write(path, test_data)
        if not success:
            raise RuntimeError(f"Failed to write test raw feature file: {error}")

        return path

    def create_test_raw_feature_hdf5(self) -> HDF5Group:
        # Build feature dataset
        feature_dataset: HDF5Dataset = self.__build_raw_feature_dataset()
        # Build feature names (rows, feature names)
        feature_names: HDF5Dataset = self.__build_raw_feature_names()
        # Build stat names (cols, stat names)
        feature_epochs: HDF5Dataset = self.__build_raw_feature_epochs()
        # Build attributes
        attributes: Dict[str, str] = self.__build_raw_attributes()
        return HDF5Group(
            name=RawFeatureFields.RAW_FEATURE.value,
            items=[feature_dataset, feature_names, feature_epochs],
            attributes=attributes,
        )

    def __build_raw_feature_dataset(self) -> HDF5Dataset:
        return HDF5Dataset(
            name=RawFeatureFields.FEATURES.value,
            data=TestConstants.FEATURE_DATA.value,
            attributes={},
        )

    def __build_raw_feature_names(self) -> HDF5Dataset:
        return HDF5Dataset(
            name=RawFeatureFields.FEATURE_NAMES.value,
            data=TestConstants.RAW_FEATURE_NAMES.value,
            attributes={},
        )

    def __build_raw_feature_epochs(self) -> HDF5Dataset:
        return HDF5Dataset(
            name=RawFeatureFields.FEATURE_EPOCHS.value,
            data=TestConstants.EPOCH_START_TIMES.value,
            attributes={},
        )

    def __build_raw_attributes(self) -> Dict[str, Any]:
        return {
            RawFeatureFields.RAW_FEATURE_IDENTIFIER.value: TestConstants.RAW_FEATURE_ID.value,
            RawFeatureFields.IMU_DATA_IDENTIFIER.value: TestConstants.FEATURE_IMU_DATA_ID.value,
            RawFeatureFields.USER_DATA_IDENTIFIER.value: TestConstants.FEATURE_USER_DATA_ID.value,
            RawFeatureFields.START_TIME.value: TestConstants.RAW_FEATURE_START_TIME.value,
            RawFeatureFields.EPOCH_LEN.value: TestConstants.RAW_FEATURE_EPOCH_LEN.value,
        }

    def create_test_raw_feature(self) -> RawFeatureSetEntry:
        return RawFeatureSetEntry(
            raw_epoch_features=self.__build_raw_epoch_feature_list(),
            metadata=self.__build_raw_metadata(),
        )

    def __build_raw_epoch_feature_list(self) -> List[RawEpochFeatures]:
        return [
            RawEpochFeatures(
                raw_features=self.__build_raw_feature_list(),
                epoch_start_time=TestConstants.RAW_FEATURE_START_TIME.value,
                epoch_end_time=TestConstants.RAW_FEATURE_START_TIME.value
                + TestConstants.RAW_FEATURE_EPOCH_LEN.value,
            ),
            RawEpochFeatures(
                raw_features=self.__build_raw_feature_list(),
                epoch_start_time=TestConstants.RAW_FEATURE_START_TIME.value + 0.1,
                epoch_end_time=TestConstants.RAW_FEATURE_START_TIME.value
                + TestConstants.RAW_FEATURE_EPOCH_LEN.value
                + 0.1,
            ),
        ]

    def __build_raw_feature_list(self) -> List[RawFeature]:
        return [
            RawFeature(
                feature_type=RawFeatureType.PLACEHOLDER,
                value=TestConstants.PLACEHOLDER_FEATURE_VALUE.value,
            ),
            RawFeature(
                feature_type=RawFeatureType.PLACEHOLDER,
                value=TestConstants.PLACEHOLDER_FEATURE_VALUE.value + 1.0,
            ),
        ]

    def __build_raw_metadata(self) -> RawFeatureSetEntryMetadata:
        return RawFeatureSetEntryMetadata(
            raw_feature_identifier=RawFeatureIdentifier(
                TestConstants.RAW_FEATURE_ID.value
            ),
            user_identifier=UserIdentifier(TestConstants.FEATURE_USER_DATA_ID.value),
            imu_data_identifier=IMUDataIdentifier(
                TestConstants.FEATURE_IMU_DATA_ID.value
            ),
            start_time=TestConstants.RAW_FEATURE_START_TIME.value,
            epoch_length=TestConstants.RAW_FEATURE_EPOCH_LEN.value,
        )


class UserDataHelper:
    """Helper class for creating test user data."""

    def create_test_user_data(self) -> UserData:
        """Create test user data model object."""
        return UserData(
            user_identifier=UserIdentifier(TestConstants.USER_DATA_ID.value),
            clinical_demographic_data=self.__build_clinical_demographic_data(),
            clinical_assessment=ClinicalAssessment(),
        )

    def create_test_user_data_file(self, path: Optional[Path] = None) -> Path:
        """Create test user data file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        # Use provided path or default
        if path is None:
            path = Path("test/test_data/user_data.json")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create and write data
        with open(path, "w") as f:
            json.dump(self.create_test_user_data_json().data, f)

        return path

    def create_test_user_data_json(self) -> JSONDictFile:
        """Create test user data JSON file."""
        return JSONDictFile(
            {
                UserDataFields.USER_DATA_IDENTIFIER.value: TestConstants.USER_DATA_ID.value
            }
        )

    def create_test_clinical_demographic_file(
        self, path: Optional[Path] = None
    ) -> Path:
        """Create test clinical demographic data file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/clinical_demographic_data.json")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.create_test_clinical_demographic_json().data, f)

        return path

    def create_test_clinical_demographic_json(self) -> JSONDictFile:
        """Create test clinical demographic JSON file."""
        return JSONDictFile(
            {
                ClinicalDemographicDataFields.NAME.value: {
                    ClinicalDemographicDataFields.VALUE.value: TestConstants.USER_NAME.value
                },
                ClinicalDemographicDataFields.AGE.value: {
                    ClinicalDemographicDataFields.VALUE.value: TestConstants.USER_AGE.value
                },
                ClinicalDemographicDataFields.SEX.value: {
                    ClinicalDemographicDataFields.VALUE.value: TestConstants.USER_SEX.value
                },
                ClinicalDemographicDataFields.WEIGHT.value: {
                    ClinicalDemographicDataFields.VALUE.value: TestConstants.USER_WEIGHT.value
                },
                ClinicalDemographicDataFields.HEIGHT.value: {
                    ClinicalDemographicDataFields.VALUE.value: TestConstants.USER_HEIGHT.value
                },
            }
        )

    def __build_clinical_demographic_data(self) -> ClinicalDemographicData:
        return ClinicalDemographicData(
            name=TestConstants.USER_NAME.value,
            age=TestConstants.USER_AGE.value,
            sex=Sex(TestConstants.USER_SEX.value),
            weight=Kilogram(TestConstants.USER_WEIGHT.value),
            height=Meter(TestConstants.USER_HEIGHT.value),
        )


class DatasetHelper:
    """Helper class for creating test dataset data."""

    def create_test_dataset(self) -> Dataset:
        """Create test dataset model object."""
        entries = [
            DatasetEntry(
                user_data_id=UserIdentifier(user_id),
                imu_data_id=IMUDataIdentifier(imu_id),
            )
            for user_id, imu_id in zip(
                TestConstants.DATASET_USER_IDS.value,
                TestConstants.DATASET_IMU_IDS.value,
            )
        ]
        return Dataset(name=TestConstants.DATASET_NAME.value, entries=entries)

    def create_test_dataset_file(self, path: Optional[Path] = None) -> Path:
        """Create test dataset file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/dataset.csv")

        path.parent.mkdir(parents=True, exist_ok=True)

        test_data = self.create_test_dataset_csv()
        df = pd.DataFrame(test_data.data)
        df.to_csv(path)

        return path

    def create_test_dataset_csv(self) -> CSVFile:
        """Create test dataset CSV file."""
        return CSVFile(
            fieldnames=[
                DatasetFields.USER_DATA_IDENTIFIER.value,
                DatasetFields.IMU_DATA_IDENTIFIER.value,
            ],
            data={
                DatasetFields.USER_DATA_IDENTIFIER.value: TestConstants.DATASET_USER_IDS.value,
                DatasetFields.IMU_DATA_IDENTIFIER.value: TestConstants.DATASET_IMU_IDS.value,
            },
        )


class FeatureSetHelper:
    """Helper class for creating test feature set data."""

    def create_test_feature_set(self) -> FeatureSet:
        """Create test feature set model object."""
        entries = [
            FeatureSetEntry(
                raw_feature_identifier=RawFeatureIdentifier(raw_id),
                aggregate_feature_identifier=AggregateFeatureIdentifier(agg_id),
            )
            for raw_id, agg_id in zip(
                TestConstants.FEATURE_SET_RAW_IDS.value,
                TestConstants.FEATURE_SET_AGG_IDS.value,
            )
        ]
        return FeatureSet(name=TestConstants.FEATURE_SET_NAME.value, entries=entries)

    def create_test_feature_set_file(self, path: Optional[Path] = None) -> Path:
        """Create test feature set file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/feature_set.csv")

        path.parent.mkdir(parents=True, exist_ok=True)

        test_data = self.create_test_feature_set_csv()
        test_data.to_csv(path)

        return path

    def create_test_feature_set_csv(self) -> CSVFile:
        """Create test feature set CSV file."""
        return CSVFile(
            fieldnames=[
                FeatureSetFields.RAW_FEATURE_IDENTIFIER.value,
                FeatureSetFields.AGGREGATE_FEATURE_IDENTIFIER.value,
            ],
            data={
                FeatureSetFields.RAW_FEATURE_IDENTIFIER.value: TestConstants.FEATURE_SET_RAW_IDS.value,
                FeatureSetFields.AGGREGATE_FEATURE_IDENTIFIER.value: TestConstants.FEATURE_SET_AGG_IDS.value,
            },
        )


class InstrumentSpecificationHelper:
    """Helper class for creating test instrument specification data."""

    def create_test_imu_specifications(self) -> IMUSpecifications:
        """Create test IMU specifications model object."""
        sensor_specs = [
            self.__build_sensor_specification(SensorType.ACCELEROMETER),
            self.__build_sensor_specification(SensorType.GYROSCOPE),
        ]
        return IMUSpecifications(
            sensor_specifications=sensor_specs,
            spec_id=InstrumentSpecificationIdentifier(TestConstants.SPEC_ID.value),
            imu_name=TestConstants.IMU_NAME.value,
        )

    def __build_sensor_specification(
        self, sensor_type: SensorType
    ) -> SensorSpecification:
        """Build sensor specification object."""
        return SensorSpecification(
            sensor_type=sensor_type,
            sensor_name=TestConstants.SENSOR_NAME.value,
            units=TestConstants.SENSOR_UNITS.value,
            range=TestConstants.SENSOR_RANGE.value,
            sensitivity=TestConstants.SENSOR_SENSITIVITY.value,
            resolution=TestConstants.SENSOR_RESOLUTION.value,
            sampling_rate=TestConstants.SENSOR_SAMPLING_RATE.value,
            noise_density=TestConstants.SENSOR_NOISE_DENSITY.value,
            bias_stability=TestConstants.SENSOR_BIAS_STABILITY.value,
            alignment_error=TestConstants.SENSOR_ALIGNMENT_ERROR.value,
            cross_axis_sensitivity=TestConstants.SENSOR_CROSS_AXIS_SENSITIVITY.value,
            power_consumption=TestConstants.SENSOR_POWER_CONSUMPTION.value,
            operating_conditions=TestConstants.SENSOR_OPERATING_CONDITIONS.value,
            physical_size=TestConstants.SENSOR_SIZE.value,
            mass=TestConstants.SENSOR_MASS.value,
        )

    def create_test_specification_file(self, path: Optional[Path] = None) -> Path:
        """Create test specification file.

        Args:
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/instrument_specification.json")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.create_test_specification_json().data, f)

        return path

    def create_test_specification_json(self) -> JSONDictFile:
        """Create test specification JSON file."""
        return JSONDictFile(
            {
                InstrumentSpecificationFields.SPEC_ID.value: TestConstants.SPEC_ID.value,
                InstrumentSpecificationFields.IMU_NAME.value: TestConstants.IMU_NAME.value,
                InstrumentSpecificationFields.SENSOR_SPECIFICATIONS.value: [
                    self.__build_sensor_specification_dict(SensorType.ACCELEROMETER),
                    self.__build_sensor_specification_dict(SensorType.GYROSCOPE),
                ],
            }
        )

    def __build_sensor_specification_dict(
        self, sensor_type: SensorType
    ) -> Dict[str, Any]:
        """Build sensor specification dictionary."""
        return {
            InstrumentSpecificationFields.SENSOR_TYPE.value: sensor_type.value,
            InstrumentSpecificationFields.SENSOR_NAME.value: TestConstants.SENSOR_NAME.value,
            InstrumentSpecificationFields.UNITS.value: TestConstants.SENSOR_UNITS.value,
            InstrumentSpecificationFields.RANGE.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_RANGE.value
            },
            InstrumentSpecificationFields.SENSITIVITY.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_SENSITIVITY.value
            },
            InstrumentSpecificationFields.RESOLUTION.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_RESOLUTION.value
            },
            InstrumentSpecificationFields.SAMPLING_RATE.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_SAMPLING_RATE.value
            },
            InstrumentSpecificationFields.NOISE_DENSITY.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_NOISE_DENSITY.value
            },
            InstrumentSpecificationFields.BIAS_STABILITY.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_BIAS_STABILITY.value
            },
            InstrumentSpecificationFields.ALIGNMENT_ERROR.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_ALIGNMENT_ERROR.value
            },
            InstrumentSpecificationFields.CROSS_AXIS_SENSITIVITY.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_CROSS_AXIS_SENSITIVITY.value
            },
            InstrumentSpecificationFields.POWER_CONSUMPTION.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_POWER_CONSUMPTION.value
            },
            InstrumentSpecificationFields.OPERATING_CONDITIONS.value: {
                InstrumentSpecificationFields.VALUE.value: TestConstants.SENSOR_OPERATING_CONDITIONS.value
            },
            InstrumentSpecificationFields.PHYSICAL_SIZE.value: TestConstants.SENSOR_SIZE.value,
            InstrumentSpecificationFields.MASS.value: TestConstants.SENSOR_MASS.value,
        }


class MappingHelper:
    """Helper class for creating test mapping data."""

    def create_test_mapping_file(
        self, source_ids: List[str], target_ids: List[str], path: Optional[Path] = None
    ) -> Path:
        """Create test mapping file.

        Args:
            source_ids (List[str]): Source identifiers
            target_ids (List[str]): Target identifiers
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/mapping.csv")

        path.parent.mkdir(parents=True, exist_ok=True)

        test_data = self.create_test_mapping_csv(source_ids, target_ids)
        test_data.to_csv(path)

        return path

    def create_test_mapping_csv(
        self, source_ids: List[str], target_ids: List[str]
    ) -> CSVFile:
        """Create test mapping CSV file."""
        return CSVFile(
            fieldnames=[
                MappingFields.SOURCE_DATA_IDENTIFIER.value,
                MappingFields.TARGET_DATA_IDENTIFIER.value,
            ],
            data={
                MappingFields.SOURCE_DATA_IDENTIFIER.value: source_ids,
                MappingFields.TARGET_DATA_IDENTIFIER.value: target_ids,
            },
        )


class RegistryHelper:
    """Helper class for creating test registry data."""

    def create_test_registry_file(
        self, ids: List[str], path: Optional[Path] = None
    ) -> Path:
        """Create test registry file.

        Args:
            ids (List[str]): Registry identifiers
            path (Optional[Path]): Path where to create the file. If None, uses default location.

        Returns:
            Path: Path to the created file
        """
        if path is None:
            path = Path("test/test_data/registry.csv")

        path.parent.mkdir(parents=True, exist_ok=True)

        test_data = self.create_test_registry_csv(ids)
        test_data.to_csv(path)

        return path

    def create_test_registry_csv(self, ids: List[str]) -> CSVFile:
        """Create test registry CSV file."""
        return CSVFile(
            fieldnames=[
                RegistryFields.DATA_IDENTIFIER.value,
                RegistryFields.DIRECTORY.value,
            ],
            data={
                RegistryFields.DATA_IDENTIFIER.value: ids,
                RegistryFields.DIRECTORY.value: TestConstants.REGISTRY_PATHS.value,
            },
        )


class FileIOHelper:
    """Helper class for file I/O testing."""

    def create_test_csv_file(self) -> CSVFile:
        """Create test CSV file object."""
        return CSVFile(
            fieldnames=TestConstants.CSV_FIELDNAMES.value,
            data=TestConstants.CSV_DATA.value,
        )

    def create_test_json_file(self) -> JSONDictFile:
        """Create test JSON file object."""
        return JSONDictFile(data=TestConstants.JSON_DATA.value)

    def create_test_hdf5_group(self) -> HDF5Group:
        """Create test HDF5 group object."""
        dataset_1 = HDF5Dataset(
            name=TestConstants.HDF5_DATASET_1_NAME.value,
            data=TestConstants.HDF5_DATA.value,
            attributes=TestConstants.HDF5_ATTRIBUTES.value,
        )
        dataset_2 = HDF5Dataset(
            name=TestConstants.HDF5_DATASET_2_NAME.value,
            data=TestConstants.HDF5_DATA.value,
            attributes=TestConstants.HDF5_ATTRIBUTES.value,
        )
        group = HDF5Group(
            name=TestConstants.HDF5_GROUP_NAME.value,
            items=[dataset_1, dataset_2],
            attributes=TestConstants.HDF5_GROUP_ATTRIBUTES.value,
        )
        return group
