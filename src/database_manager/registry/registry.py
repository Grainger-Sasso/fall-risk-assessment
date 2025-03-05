from pathlib import Path
from typing import Dict, Type

from src.identifiers.identifier import Identifier


class Registry:
    """
    Class for data registries - mappings between string data identifiers and file paths
    """

    def __init__(self, registry: Dict[str, Path], id_type: Type[Identifier]):
        self._registry: Dict[str, Path] = registry
        self._id_type: Type[Identifier] = id_type

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

    @property
    def id_type(self) -> Type[Identifier]:
        return self._id_type
