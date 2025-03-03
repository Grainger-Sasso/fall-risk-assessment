from pathlib import Path
from typing import Dict, Generic, TypeVar

from src.database_manager.registries.registry import Registry
from src.identifiers.identifier import Identifier

T = TypeVar("T", bound=Identifier)


class RegistryManager(Generic[T]):
    """Manages access to registry data"""

    def __init__(self, registry: Registry):
        self._registry = registry

    def get_path(self, identifier: T) -> Path:
        """Get file path for a given identifier

        Args:
            identifier: Unique identifier for the data

        Returns:
            Path to the data file

        Raises:
            KeyError: If identifier not found in registry
        """
        if identifier not in self._registry.registry:
            raise KeyError(f"No path found for identifier: {identifier}")
        return self._registry.registry[identifier]

    def get_all_paths(self) -> Dict[T, Path]:
        """Get all paths in the registry

        Returns:
            Dictionary mapping identifiers to paths
        """
        return self._registry.registry
