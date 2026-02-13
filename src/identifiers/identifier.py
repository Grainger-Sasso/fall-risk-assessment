import uuid
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

T = TypeVar("T", bound="Identifier")


class Identifier(ABC):
    """
    Abstract base class for representing unique identifiers.
    Subclasses should implement specific types of identifiers.
    """

    def __init__(self, value: str):
        """
        Initialize the Identifier with a value.

        :param value: A string representing the unique identifier.
        """
        self.value = value

    @property
    def value(self) -> str:
        """
        Get the value of the identifier.

        :return: The string value of the identifier.
        """
        return self._value

    @value.setter
    def value(self, value: str):
        """
        Set the value of the identifier.

        :param value: A string representing the unique identifier.
        """
        if not self.validate(value):
            raise ValueError("Invalid value.")
        self._value = value

    @abstractmethod
    def validate(self, value: str) -> bool:
        """
        Validate the identifier. This method should be implemented by subclasses
        to provide specific validation logic for the type of identifier.

        :return: True if the identifier is valid, False otherwise.
        """
        pass

    def __str__(self):
        """
        Return the string representation of the identifier.

        :return: The string value of the identifier.
        """
        return self.value

    def __eq__(self, other):
        """
        Check if two identifiers are equal based on their values.

        :param other: Another Identifier object to compare with.
        :return: True if the values are equal, False otherwise.
        """
        if isinstance(other, Identifier):
            return self.value == other.value
        return False

    def __hash__(self):
        """
        Return the hash value of the identifier based on its value.

        :return: The hash value of the identifier.
        """
        return hash(self.value)


class IdentifierGenerator(Generic[T]):
    """
    Generic identifier generator that creates unique identifiers
    of a given Identifier subtype with a specified prefix.
    """

    def __init__(self, prefix: str, id_class: Type[T]):
        self.prefix: str = prefix
        self.id_class: Type[T] = id_class

    def generate_identifier(self) -> T:
        id_value = f"{self.prefix}_{uuid.uuid4()}"
        return self.id_class(id_value)
