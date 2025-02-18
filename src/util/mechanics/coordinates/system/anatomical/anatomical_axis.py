from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)


class AnatomicalAxis:
    def __init__(self, name: AnatomicalCoordinateSystem):
        self._name = name

    @property
    def name(self) -> AnatomicalCoordinateSystem:
        return self._name
