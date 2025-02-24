from src.util.mechanics.units.unit import Unit


class Acceleration(Unit):
    """
    Represents acceleration in the SI system.
    """

    def __init__(self, value: float):
        super().__init__(value, "meter/second**2")
