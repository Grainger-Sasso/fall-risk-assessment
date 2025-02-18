from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


class SensorAxis:
    def __init__(self, name: SensorCoordinateSystem):
        self._name = name

    @property
    def name(self) -> SensorCoordinateSystem:
        return self._name
