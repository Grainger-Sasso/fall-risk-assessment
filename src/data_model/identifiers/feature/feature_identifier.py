from src.data_model.identifiers.identifier import Identifier


class FeatureIdentifier(Identifier):
    """
    Represents unique raw feature identifiers.
    """

    def __init__(self, value: str):
        """
        Initialize the Identifier with a value.

        :param value: A string representing the unique identifier.
        """
        super().__init__(value)

    def validate(self, value: str) -> bool:
        """
        Dummy validation on setter input.


        Args:
            value (str): Value of identifier

        Returns:
            bool: Input value is valid identifier
        """
        return True
