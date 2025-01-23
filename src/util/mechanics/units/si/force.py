from src.util.mechanics.units import Unit

class Force(Unit):
    """Represents force in the SI system."""
    def __init__(self, value: float):
        super().__init__(value, "kilogram*meter/second**2")
