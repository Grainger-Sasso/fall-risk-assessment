from src.util.mechanics.units import Unit

class Joule(Unit):
    """
    Represents energy in the SI system.
    """
    def __init__(self, value: float):
        super().__init__(value, "kilogram*meter**2/second**2")
