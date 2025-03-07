from typing import Dict, Type

from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier


class RegistryManager(DataAccessManager[Registry]):
    """Manages access to registries"""

    def __init__(self, providers: Dict[Type[Identifier], Registry]):
        super().__init__(providers)
