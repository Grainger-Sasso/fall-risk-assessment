from src.util.mechanics.units import Unit

class Time(Unit):
    """Represents time in the SI system."""
    def __init__(self, value: float):
        super().__init__(value, "second")
