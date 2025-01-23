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
    
















# Example usage
if __name__ == "__main__":
    mass = Mass(5)
    print(f"Mass: {mass}")

    length = Length(10)
    print(f"Length: {length}")

    time = Time(60)
    print(f"Time: {time}")

    velocity = Velocity(30)
    print(f"Velocity: {velocity}")

    acceleration = Acceleration(9.8)
    print(f"Acceleration: {acceleration}")

    force = Force(50)
    print(f"Force: {force}")

    energy = Energy(1000)
    print(f"Energy: {energy}")

    power = Power(500)
    print(f"Power: {power}")
