from abc import ABC
from typing import Dict, Generic, Type, TypeVar

from src.data_io.import_export.importers.importer import Importer
from src.identifiers.identifier import Identifier

T = TypeVar("T")


class DataAccessManager(Generic[T], ABC):
    """
    Manager responsible for providing data access providers
    """

    def __init__(self, providers: Dict[Type[Identifier], T]):
        self._providers: Dict[Type[Identifier], T] = providers

    @property
    def providers(self):
        return self._providers

    def get_provider(self, data_type: Type[Identifier]) -> T:
        if data_type not in self.providers:
            raise KeyError(f"Unable to resolve provider from ID: {data_type}")
        return self.providers[data_type]
