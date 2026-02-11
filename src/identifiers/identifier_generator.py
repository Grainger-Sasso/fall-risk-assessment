import uuid
from typing import Generic, Type, TypeVar

from src.identifiers.identifier import Identifier

T = TypeVar("T", bound=Identifier)


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
