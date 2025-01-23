from src.util.mechanics.units import Unit

class Length(Unit):
    """Represents length in the SI system."""
    def __init__(self, value: float):
        super().__init__(value, "meter")
