from src.util.mechanics.units.unit import Unit


class Meter(Unit):
    """
    Represents length in the SI system.
    """

    def __init__(self, value: float):
        super().__init__(value, "meter")
