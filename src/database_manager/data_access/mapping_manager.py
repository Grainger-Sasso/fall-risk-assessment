from typing import Dict, Generic, List, TypeVar

from src.identifiers.identifier import Identifier

S = TypeVar("S", bound=Identifier)
T = TypeVar("T", bound=Identifier)


class MappingManager(Generic[S, T]):
    """Manages relationships between different identifier types"""

    def __init__(self, mapping: Dict[S, T]):
        self._mapping = mapping

    def get_target_id(self, source_id: S) -> T:
        """Get target identifier for a given source identifier

        Args:
            source_id: Source identifier

        Returns:
            Corresponding target identifier

        Raises:
            KeyError: If source_id not found in mapping
        """
        if source_id not in self._mapping:
            raise KeyError(f"No mapping found for source id: {source_id}")
        return self._mapping[source_id]

    def get_target_ids(self, source_ids: List[S]) -> List[T]:
        """Get target identifiers for a list of source identifiers

        Args:
            source_ids: List of source identifiers

        Returns:
            List of corresponding target identifiers

        Raises:
            KeyError: If any source_id not found in mapping
        """
        return [self.get_target_id(id) for id in source_ids]
