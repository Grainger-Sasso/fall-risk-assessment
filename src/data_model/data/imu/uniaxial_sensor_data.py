from dataclasses import dataclass
import numpy as np
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis


@dataclass
class UniaxialSensorData:
    """
    Stores data for a bodily-worn IMU sensor.
    """

    anatomical_axis: AnatomicalAxis
    sensor_axis: SensorAxis
    data: np.ndarray

    @property
    def data(self) -> np.ndarray:
        """
        Getter for data. Returns the sensor data as a numpy array.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """
        Setter for data. Validates that the input is a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("data must be a numpy array")
        self._data = value
