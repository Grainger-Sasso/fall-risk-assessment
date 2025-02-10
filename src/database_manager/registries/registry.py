from abc import ABC
from pathlib import Path
from typing import Dict


from src.identifiers.identifier import Identifier


class Registry(ABC):
    """
    Abstract class for data registries - mappings between data identifiers and files
    Args:
        ABC (_type_): _description_
    """

    def __init__(self, registry: Dict[Identifier:Path]):
        self.registry: Dict[Identifier:Path] = registry

    @property
    def registry(self) -> str:
        """
        Gets the registry.

        :return: The registry.
        """
        return self._registry

    @registry.setter
    def registry(self, registry: Dict[Identifier:Path]):
        """
        Set the value of the registry.

        :param registry: A string representing the unique identifier.
        """
        self._registry = registry
