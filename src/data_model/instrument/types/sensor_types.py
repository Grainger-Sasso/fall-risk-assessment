from enum import Enum


class SensorTypes(Enum):
    """
    Collection of sensor types
    """
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
