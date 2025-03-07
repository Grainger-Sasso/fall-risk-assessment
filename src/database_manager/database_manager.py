from pathlib import Path
from typing import Any

from src.data_io.import_export.importers.importer import Importer
from src.database_manager.data_access.export_manager import ExportManager
from src.database_manager.data_access.import_manager import ImportManager
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier


class DatabaseManager:
    def __init__(
        self,
        registry_manager: RegistryManager,
        mapping_manager: MappingManager,
        import_manager: ImportManager,
        export_manager: ExportManager,
    ):
        self._registry_manager: RegistryManager = registry_manager
        self._mapping_manager: MappingManager = mapping_manager
        self._import_manager: ImportManager = import_manager
        self._export_manager: ExportManager = export_manager

    ### Registry Methods ###
    def get_data(self, identifier: Identifier) -> Any:
        # Get corresponding registry and importer
        registry: Registry = self._registry_manager.get_registry(identifier)
        importer: Importer = self._import_manager.get_importer(type(identifier))
        # Get path of data from registry using provided ID
        path: Path = registry.get_path(identifier)

        return self._data_loader.load(id, path)

    # def update_data(self, id: Identifier, data: Any):
    #     path = self._registry_manager.get_path(id)
    #     # Delete the data at this path
    #     # Write the new data to this path

    # def delete_data(self, id: Identifier):
    #     path = self._registry_manager.get_path(id)
    #     # Delete the data at this path
    #     # Remove the entry from the registry

    # def register_data(self, data: Any):
    #     #
    #     # export
    #     pass
