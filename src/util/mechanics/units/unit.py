import pint

# Initialize the Pint UnitRegistry
ureg = pint.UnitRegistry()

class Unit:
    """
    Base class for all units. Provides a common interface for accessing units.
    """
    def __init__(self, value: float, unit: str):
        self.value = ureg.Quantity(value, unit)

    def to(self, unit: str):
        """Convert the current value to another compatible unit."""
        return self.value.to(unit)

    def __repr__(self):
        return f"{self.value}"
