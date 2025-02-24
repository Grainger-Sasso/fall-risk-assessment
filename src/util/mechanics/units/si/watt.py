from src.util.mechanics.units.unit import Unit


class Watt(Unit):
    """
    Represents power in the SI system.
    """

    def __init__(self, value: float):
        super().__init__(value, "kilogram*meter**2/second**3")
