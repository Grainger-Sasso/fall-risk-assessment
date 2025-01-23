from src.util.mechanics.units import Unit

class Mass(Unit):
    """Represents mass in the SI system."""
    def __init__(self, value: float):
        super().__init__(value, "kilogram")
