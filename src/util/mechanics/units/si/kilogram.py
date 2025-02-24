from src.util.mechanics.units.unit import Unit


class Kilogram(Unit):
    """
    Represents mass in the SI system.
    """

    def __init__(self, value: float):
        super().__init__(value, "kilogram")
