from typing import Dict, Type

from src.database_manager.mapping.mapping import Mapping
from src.identifiers.identifier import Identifier


class MappingManager:
    """Manages relationships between different identifier types"""

    def __init__(self, mappings: Dict[Type[Identifier], Mapping]):
        self._mappings: Dict[Type[Identifier], Mapping] = mappings

    @property
    def mappings(self) -> Dict[Type[Identifier], Mapping]:
        return self._mappings

    def get_mapping(self, source_id: Identifier) -> Mapping:
        """Get target identifier for a given source identifier

        Args:
            source_id: Source identifier

        Returns:
            Corresponding target identifier

        Raises:
            KeyError: If source_id not found in mapping
        """
        if type(source_id) not in self.mappings:
            raise KeyError(f"Unable to resolve mapping from ID: {source_id}")
        return self.mappings[type(source_id)]

