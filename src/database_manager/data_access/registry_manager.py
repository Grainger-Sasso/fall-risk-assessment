from pathlib import Path
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

    def get_path(self, identifier: Identifier) -> Path:
        """Get file path for a given identifier

        Args:
            identifier: Unique identifier for the data

        Returns:
            Path to the data file

        Raises:
            KeyError: If identifier not found in registry
        """
        if type(identifier) not in self._registry_map:
            raise KeyError(f"Unable to resolve registry from ID: {identifier}")
        registry = self._registry_map[type(identifier)]
        if identifier.value not in registry.registry.keys():
            raise KeyError(f"Unable to resolve path of ID: {identifier}")
        return registry.registry[identifier.value]
