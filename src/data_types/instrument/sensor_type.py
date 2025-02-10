from enum import Enum


class SensorType(Enum):
    """
    Collection of sensor types
    """

    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
