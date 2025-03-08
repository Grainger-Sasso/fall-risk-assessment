from enum import Enum


class IMUDataFields(Enum):
    """Fields used in imu data files"""

    IMU_DATA = "imu_data"
    SENSOR_DATA = "sensor_data"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    DATA = "data"
    TIME = "time"
    SENSOR_TYPE = "sensor_type"
    ORIENTATION_MAP_SENSOR = "orientation_map_sensor"
    ORIENTATION_MAP_ANATOMICAL = "orientation_map_anatomical"
    SENSOR_AXIS_X = "sensor_axis_x"
    SENSOR_AXIS_Y = "sensor_axis_y"
    SENSOR_AXIS_Z = "sensor_axis_z"
    ANATOMICAL_AXIS_MEDIOLATERAL = "anatomical_axis_mediolateral"
    ANATOMICAL_AXIS_ANTEROPOSTERIOR = "anatomical_axis_anteroposterior"
    ANATOMICAL_AXIS_VERTICAL = "anatomical_axis_vertical"
    SAMPLING_RATE = "sampling_rate"
    UNIT = "unit"
    AXIS_NAMES = "axis_names"
    IMU_DATA_IDENTIFIER = "imu_data_identifier"
    USER_IDENTIFIER = "user_identifier"
    INSTRUMENT_NAME = "instrument_name"
    SERIAL_NUMBER = "serial_number"
