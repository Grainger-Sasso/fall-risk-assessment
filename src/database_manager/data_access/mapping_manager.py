from typing import Dict, Type

from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.database_manager.mapping.mapping import Mapping
from src.identifiers.identifier import Identifier


class MappingManager(DataAccessManager[Mapping]):
    """Manages access to mappings"""

    def __init__(self, providers: Dict[Type[Identifier], Mapping]):
        super().__init__(providers)
