import pint

# Initialize the Pint UnitRegistry
ureg = pint.UnitRegistry()


class Unit:
    """
    Base class for all units. Provides a common interface for accessing units.
    """

    def __init__(self, value: float, unit: str):
        self._value = value
        self._unit = unit  # Set unit once and make it immutable
        self._quantity = ureg.Quantity(value, unit)

    @property
    def value(self) -> float:
        """Returns the value of the unit."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        """Sets the value of the unit."""
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a number.")
        self._value = value
        self._quantity = ureg.Quantity(value, self._unit)

    @property
    def unit(self) -> str:
        """Returns the unit of the quantity (immutable)."""
        return self._unit

    def to(self, unit: str):
        """Convert the current value to another compatible unit."""
        return self._quantity.to(unit)

    def __repr__(self):
        return f"{self._quantity}"
