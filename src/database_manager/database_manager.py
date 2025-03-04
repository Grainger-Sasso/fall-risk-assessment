from pathlib import Path
from typing import Any

from src.database_manager.data_access.data_loader import DataLoader
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.registry_manager import RegistryManager
from src.identifiers.identifier import Identifier


class DatabaseManager:
    def __init__(
        self,
        registry_manager: RegistryManager,
        mapping_manager: MappingManager,
        data_loader: DataLoader,
    ):
        self._registry_manager: RegistryManager = registry_manager
        self._mapping_manager: MappingManager = mapping_manager
        # self._import_manager
        # self._export_manager

    ### Registry Methods ###
    def get_data(self, id: Identifier) -> Any:
        path: Path = self._registry_manager.get_path(id)
        # Import data
        return self._data_loader.load(id, path)

    def update_data(self, id: Identifier, data: Any):
        path = self._registry_manager.get_path(id)
        # Delete the data at this path
        # Write the new data to this path

    def delete_data(self, id: Identifier):
        path = self._registry_manager.get_path(id)
        # Delete the data at this path
        # Remove the entry from the registry

    def register_data(self, data: Any):
        #
        # export 
        pass
