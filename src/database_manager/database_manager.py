from pathlib import Path
from typing import Tuple, Type, TypeVar

from src.data_io.import_export.importers.importer import Importer
from src.database_manager.data_access.export_manager import ExportManager
from src.database_manager.data_access.import_manager import ImportManager
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier

T = TypeVar("T")


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
    def get_data(self, identifier: Identifier) -> T:
        data_type: Type[Identifier] = type(identifier)
        # Get corresponding registry and importer
        registry: Registry = self._registry_manager.get_provider(data_type)
        importer: Importer = self._import_manager.get_provider(data_type)
        # Get path of data from registry using provided ID
        path: Path = registry.get_path(identifier)
        # Import data from path
        return importer.import_data(path)

    def write_data(self, data: T, identifier: Identifier) -> Tuple[bool, str]:
        # Get the corresponding registry
        pass

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

    ### Mapping Methods ###
    # TODO
    # get_all_imu_data_for_user [1:n mapping]
    # get_all_aggregate_features_for_raw_feature [1:n mapping]
    # get_all_raw_features_for_imu_data [1:n mapping]
    # get_insturment_spec_for_instrument [1:1 mapping]
