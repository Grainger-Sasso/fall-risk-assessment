from src.util.mechanics.units.unit import Unit


class Velocity(Unit):
    """
    Represents velocity in the SI system.
    """

    def __init__(self, value: float):
        super().__init__(value, "meter/second")
