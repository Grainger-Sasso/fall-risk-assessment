from src.util.mechanics.units import Unit

class Power(Unit):
    """Represents power in the SI system."""
    def __init__(self, value: float):
        super().__init__(value, "kilogram*meter**2/second**3")
