from pathlib import Path
from typing import Dict, Type

from src.identifiers.identifier import Identifier


class Registry:
    """
    Class for data registries - mappings between string data identifiers and file paths
    """

    def __init__(
        self, registry: Dict[str, Path], id_type: Type[Identifier], subdir_path: Path
    ):
        self._registry: Dict[str, Path] = registry
        self._id_type: Type[Identifier] = id_type
        # Path to subdir containing mapping
        self._path: Path = subdir_path

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

    @property
    def path(self) -> Path:
        return self._path

    def get_path(self, identifier: Identifier) -> Path:
        if identifier.value not in self.registry.keys():
            raise KeyError(f"Unable to resolve path from ID: {identifier}")
        return self.registry[identifier.value]
