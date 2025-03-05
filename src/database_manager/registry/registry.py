from pathlib import Path
from typing import Dict


class Registry:
    """
    Class for data registries - mappings between string data identifiers and file paths
    """

    def __init__(self, registry: Dict[str, Path]):
        self._registry: Dict[str, Path] = registry

    @property
    def registry(self) -> Dict[str, Path]:
        """
        Gets the registry.

        :return: The registry.
        """
        return self._registry

    @registry.setter
    def registry(self, registry: Dict[str, Path]):
        """
        Set the value of the registry.

        :param registry: A string representing the unique identifier.
        """
        self._registry = registry
