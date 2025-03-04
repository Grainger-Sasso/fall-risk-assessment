from typing import Dict, Generic, TypeVar

from src.identifiers.identifier import Identifier

S = TypeVar("S", bound=Identifier)
T = TypeVar("T", bound=Identifier)


class Mapping(Generic[S, T]):
    """Base class for identifier mappings"""

    def __init__(self, map: Dict[S, T]):
        self._map: Dict[S, T] = map

    @property
    def map(self) -> Dict[S, T]:
        """Get the mapping dictionary

        Returns:
            Dictionary mapping source to target identifiers
        """
        return self._map

    @map.setter
    def map(self, map: Dict[S, T]) -> None:
        """Set the mapping dictionary

        Args:
            map: Dictionary mapping source to target identifiers
        """
        self._map = map
