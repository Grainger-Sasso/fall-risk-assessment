from dataclasses import dataclass, field


@dataclass
class IMUIdentifier:
    """
    Structured, human-readable identifier for IMU device
    """

    name: str
    serial_number: str

    @property
    def identifier(self) -> str:
        """
        Returns the device identifier.
        """
        return f"{self._name}-{self._serial_number}"
