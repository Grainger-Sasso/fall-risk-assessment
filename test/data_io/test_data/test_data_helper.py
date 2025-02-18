import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_types.instrument.sensor_type import SensorType


class TestDataHelper:
    """Helper class for managing test data files and HDF5 operations.

    This class provides utilities for creating and managing HDF5 test files,
    including standard test data structures and file operations.

    Attributes:
        BASE_TEST_DATA_DIR (Path): Base directory containing test data templates
        TEMP_TEST_DATA_DIR (Path): Directory for temporary test files
        writer (JHDF5FileWriter): HDF5 file writer instance
    """

    # Base directory for test data templates
    BASE_TEST_DATA_DIR = Path(__file__).parent / "data"

    # Directory for temporary test files
    TEMP_TEST_DATA_DIR = Path(__file__).parent / "temp"

    # Test data file paths
    IMU_DATA_PATH = BASE_TEST_DATA_DIR / "imu" / "empty_imu_data.h5"
    RAW_FEATURE_PATH = BASE_TEST_DATA_DIR / "features" / "raw" / "empty_raw_feature.h5"
    AGGREGATE_FEATURE_PATH = (
        BASE_TEST_DATA_DIR / "features" / "aggregate" / "empty_aggregate_feature.h5"
    )

    def __init__(self):
        """Initialize the test data helper."""
        self.writer = HDF5FileWriter()
        self._ensure_dirs_exist()
        self._create_test_data_files()

    def _ensure_dirs_exist(self) -> None:
        """Ensure all required directories exist."""
        # Create base test data directories
        (self.BASE_TEST_DATA_DIR / "imu").mkdir(parents=True, exist_ok=True)
        (self.BASE_TEST_DATA_DIR / "features" / "raw").mkdir(
            parents=True, exist_ok=True
        )
        (self.BASE_TEST_DATA_DIR / "features" / "aggregate").mkdir(
            parents=True, exist_ok=True
        )

        # Create temp directory
        self.TEMP_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _create_test_data_files(self) -> None:
        """Create empty test data files if they don't exist."""
        if not self.IMU_DATA_PATH.exists():
            self._create_empty_file(self.IMU_DATA_PATH)
        if not self.RAW_FEATURE_PATH.exists():
            self._create_empty_file(self.RAW_FEATURE_PATH)
        if not self.AGGREGATE_FEATURE_PATH.exists():
            self._create_empty_file(self.AGGREGATE_FEATURE_PATH)

    def _create_empty_file(self, file_path: Path) -> None:
        """Create an empty HDF5 file with basic structure.

        Args:
            file_path (Path): Path where file should be created
        """
        group = HDF5Group()
        group.name = "root"
        group.attributes = {}
        group.items = []

        success, error = self.writer.write(file_path, group)
        if not success:
            raise ValueError(f"Failed to create test file {file_path}: {error}")

    def get_temp_copy(self, template_path: Path) -> Path:
        """Create a temporary copy of a test data file.

        Args:
            template_path (Path): Path to template file to copy

        Returns:
            Path: Path to temporary copy

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        temp_path = self.TEMP_TEST_DATA_DIR / template_path.name
        shutil.copy2(template_path, temp_path)
        return temp_path

    def create_empty_imu_data_file(self) -> Path:
        """Create a temporary copy of empty IMU data test file.

        Returns:
            Path: Path to temporary test file
        """
        return self.get_temp_copy(self.IMU_DATA_PATH)

    def create_empty_raw_feature_file(self) -> Path:
        """Create a temporary copy of empty raw feature test file.

        Returns:
            Path: Path to temporary test file
        """
        return self.get_temp_copy(self.RAW_FEATURE_PATH)

    def create_empty_aggregate_feature_file(self) -> Path:
        """Create a temporary copy of empty aggregate feature test file.

        Returns:
            Path: Path to temporary test file
        """
        return self.get_temp_copy(self.AGGREGATE_FEATURE_PATH)

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
            print(f"\nFull error from writer:")
            print(error)
            raise ValueError(f"Failed to create test file: {error}")

        return file_path

    def cleanup_test_files(self) -> None:
        """Remove all temporary test files."""
        if self.TEMP_TEST_DATA_DIR.exists():
            shutil.rmtree(self.TEMP_TEST_DATA_DIR)
        self.TEMP_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def create_test_imu_data_file(self) -> Path:
        """Create a test IMU data file.

        Returns:
            Path: Path to created test file
        """
        imu_data_group = HDF5Group(
            name=IMUDataFields.IMU_DATA.value,
            attributes=self.__build_test_imu_data_attributes(),
            items=self.__build_sensor_data_group(),
        )
        return self.create_test_file("test_imu_data.h5", imu_data_group)

    def __build_test_imu_data_attributes(self) -> Dict[str, Any]:
        """Build test IMU data attributes."""
        test_attributes = {}
        test_attributes[IMUDataFields.IMU_DATA_IDENTIFIER.value] = (
            "test_imu_data_identifier"
        )
        test_attributes[IMUDataFields.INSTRUMENT_IDENTIFIER.value] = (
            "test_instrument_identifier"
        )
        test_attributes[IMUDataFields.INSTRUMENT_NAME.value] = "test_instrument_name"
        test_attributes[IMUDataFields.SERIAL_NUMBER.value] = 1234567890
        return test_attributes

    def __build_sensor_data_group(self) -> List[HDF5Group]:
        """Build test IMU data items."""
        sensor_data_group = HDF5Group(
            name=IMUDataFields.SENSOR_DATA.value, items=[], attributes={}
        )
        items = []
        sensors = [
            (SensorType.ACCELEROMETER, IMUDataFields.ACCELEROMETER),
            (SensorType.GYROSCOPE, IMUDataFields.GYROSCOPE),
        ]
        for sensor_type, sensor_name in sensors:
            items.append(self.__build_sensor_data_group_item(sensor_type, sensor_name))
        sensor_data_group.items = items
        return [sensor_data_group]

    def __build_sensor_data_group_item(
        self, sensor_type: SensorType, sensor_name: IMUDataFields
    ) -> HDF5Group:
        """Build sensor group data item"""
        # Build time dataset
        time_dataset = HDF5Dataset(
            name=IMUDataFields.TIME.value, data=[0.0 for i in range(10)], attributes={}
        )
        # Build imu data dataset
        imu_data_dataset = HDF5Dataset(
            name=IMUDataFields.DATA.value,
            data=[[0.0 for i in range(10)] for j in range(3)],
            attributes={},
        )
        # Build sensor data group attributes
        sensor_data_group_attr: Dict[str, Any] = {
            IMUDataFields.SENSOR_TYPE.value: sensor_type.value,
            IMUDataFields.SENSOR_TO_ANATOMICAL_AXIS_MAP.value: {
                IMUDataFields.SENSOR_AXIS_X.value: IMUDataFields.ANATOMICAL_AXIS_ANTEROPOSTERIOR.value,
                IMUDataFields.SENSOR_AXIS_Y.value: IMUDataFields.ANATOMICAL_AXIS_MEDIOLATERAL.value,
                IMUDataFields.SENSOR_AXIS_Z.value: IMUDataFields.ANATOMICAL_AXIS_VERTICAL.value,
            },
            IMUDataFields.SAMPLING_RATE: 100.0,
            IMUDataFields.UNIT: "test_unit",
            IMUDataFields.SENSOR_AXIS_NAMES: [
                IMUDataFields.SENSOR_AXIS_X,
                IMUDataFields.SENSOR_AXIS_Y,
                IMUDataFields.SENSOR_AXIS_Z,
            ],
        }
        # Build sensor data group
        return HDF5Group(
            name=sensor_name.value,
            items=[time_dataset, imu_data_dataset],
            attributes=sensor_data_group_attr,
        )


def main():
    """Run test data generation."""
    # Create helper instance
    helper = TestDataHelper()

    try:
        # Generate test IMU data file
        file_path = helper.create_test_imu_data_file()
        print(f"Successfully created test IMU data file at: {file_path}")
    except Exception as e:
        import traceback

        print(f"Error creating test file: {e}")
        print("\nFull traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
