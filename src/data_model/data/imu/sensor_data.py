from dataclasses import dataclass, field
from typing import List, Dict

from data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_model.data.imu.sensor.sensor_metadata import SensorMetadata
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis


@dataclass
class SensorData:
    """
    Stores data from bodily-worn sensor.
    """

    data: List[UniaxialSensorData]
    metadata: SensorMetadata
    _anatomical_axis_map: Dict[AnatomicalAxis, UniaxialSensorData] = field(
        init=False, repr=False
    )
    _sensor_axis_map: Dict[SensorAxis, UniaxialSensorData] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        # Create the axis maps during initialization
        self._anatomical_axis_map = {axis.anatomical_axis: axis for axis in self.data}
        self._sensor_axis_map = {axis.sensor_axis: axis for axis in self.data}

    def _get_data_by_axis(self, axis_map: Dict, axis_value) -> UniaxialSensorData:
        """
        Generalized helper function for retrieving data by anatomical or sensor axis.
        """
        if axis_value not in axis_map:
            raise ValueError(f"{axis_value} not present in data")
        return axis_map[axis_value]

    def get_data_by_anatomical_axis(
        self, anatomical_axis: AnatomicalAxis
    ) -> UniaxialSensorData:
        """
        Retrieve data corresponding to a specific anatomical axis.
        """
        return self._get_data_by_axis(self._anatomical_axis_map, anatomical_axis)

    def get_data_by_sensor_axis(self, sensor_axis: SensorAxis) -> UniaxialSensorData:
        """
        Retrieve data corresponding to a specific sensor axis.
        """
        return self._get_data_by_axis(self._sensor_axis_map, sensor_axis)
