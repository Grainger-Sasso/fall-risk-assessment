from src.identifiers.identifier import Identifier


class InstrumentIdentifier(Identifier):
    """
    Represents unique instrument identifiers.
    """

    def __init__(self, name: str, serial_number: str):
        """
        Initialize the InstrumentIdentifier with a name and serial number.

        Args:
            name (str): The name of the instrument. NOTE - the name is intended to be camel-case
            serial_number (str): The serial number of the instrument.
        """
        # Combine name and serial number into a single value
        value = f"{name}_{serial_number}"
        # Call the parent class's initializer with the combined value
        super().__init__(value)

    def validate(self, value: str) -> bool:
        """
        Validate the instrument identifier. For this example, a valid identifier
        must contain a name and serial number separated by an underscore.

        Args:
            value (str): The value to validate.

        Returns:
            bool: True if the identifier is valid, False otherwise.
        """
        # Check if the value contains exactly one underscore
        print("################")
        print(value)
        parts = value.split("_")
        return len(parts) == 2 and all(part.strip() != "" for part in parts)
