class IMUIdentifier:
    """
    Structured, human-readable identifier for IMU device
    """

    def __init__(self, name: str, serial_number: str):
        self._name: str = name
        self._serial_number: str = serial_number
        self._identifier: str = f"{self._name}-{self._serial_number}"

    # Getter for name
    @property
    def name(self) -> str:
        return self._name

    # Getter for serial_number
    @property
    def serial_number(self) -> str:
        return self._serial_number

    # Getter for identifier
    @property
    def identifier(self) -> str:
        return self._identifier
