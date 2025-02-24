import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_io.model_fields.data.user.clinical_demographic_data_fields import (
    ClinicalDemographicDataFields,
)
from src.data_io.model_fields.data.user.user_data_fields import UserDataFields
from src.data_io.model_fields.features.aggregate.aggregate_feature_fields import (
    AggregateFeatureFields,
)
from src.data_io.model_fields.features.raw.raw_feature_fields import RawFeatureFields
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

    def create_test_imu_data_file(self) -> Path:
        """Create a test IMU data file.

        Returns:
            Path: Path to created test file
        """
        imu_data_group: HDF5Group = self.create_test_imu_data_hdf5()
        return self.test_data_helper.create_test_file(
            TestConstants.IMU_FILE_NAME.value, imu_data_group
        )

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
    def __init__(self):
        self.test_data_helper = TestDataHelper()

    ############### AGGREGATE ###############
    def create_test_aggregate_feature_file(self):
        """Create a test aggregate feature file.

        Returns:
            Path: Path to created test file
        """
        aggregate_feature: HDF5Group = self.create_test_aggregate_feature_hdf5()
        return self.test_data_helper.create_test_file(
            TestConstants.AGGREGATE_FILE_NAME.value, aggregate_feature
        )

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
    def create_test_raw_feature_file(self):
        """Create a test raw feature file.

        Returns:
            Path: Path to created test file
        """
        raw_feature: HDF5Group = self.create_test_raw_feature_hdf5()
        return self.test_data_helper.create_test_file(
            TestConstants.RAW_FILE_NAME.value, raw_feature
        )

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

    def create_test_user_data_json(self) -> JSONDictFile:
        """Create test user data JSON file."""
        return JSONDictFile(
            {
                UserDataFields.USER_DATA_IDENTIFIER.value: TestConstants.USER_DATA_ID.value
            }
        )

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
