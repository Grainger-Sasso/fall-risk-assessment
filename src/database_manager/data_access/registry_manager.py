from typing import Dict, Type

from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier


class RegistryManager:
    """Manages access to registry data"""

    def __init__(self, registry_map: Dict[Type[Identifier], Registry]):
        self._registry_map: Dict[Type[Identifier], Registry] = registry_map

    @property
    def registry_map(self) -> Dict[Type[Identifier], Registry]:
        return self._registry_map

    def get_registry(self, identifier: Identifier) -> Registry:
        """Get file registry for a given identifier

        Args:
            identifier: Unique identifier for the data

        Returns:
            Registry

        Raises:
            KeyError: If identifier not found in registry
        """
        if type(identifier) not in self.registry_map:
            raise KeyError(f"Unable to resolve registry from ID: {identifier}")
        return self.registry_map[type(identifier)]
