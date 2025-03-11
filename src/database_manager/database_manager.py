from pathlib import Path
from typing import Dict, List, Type, TypeVar

from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.importers.importer import Importer
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.database_manager.data_access.export_manager import ExportManager
from src.database_manager.data_access.import_manager import ImportManager
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.output_directory_manager import (
    OutputDirectoryManager,
)
from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier

S = TypeVar("S")
T = TypeVar("T")


class DatabaseManager:
    def __init__(
        self,
        registry_manager: RegistryManager,
        mapping_manager: MappingManager,
        import_manager: ImportManager,
        export_manager: ExportManager,
        output_dir_manager: OutputDirectoryManager,
    ):
        self.registry_manager: RegistryManager = registry_manager
        self.mapping_manager: MappingManager = mapping_manager
        self.import_manager: ImportManager = import_manager
        self.export_manager: ExportManager = export_manager
        self.output_dir_manager: OutputDirectoryManager = output_dir_manager

    ### Registry Methods ###
    def import_data(self, identifier: Identifier) -> T:
        data_type: Type[Identifier] = type(identifier)
        # Get corresponding registry and importer
        registry: Registry = self.registry_manager.get_provider(data_type)
        importer: Importer = self.import_manager.get_provider(data_type)
        # Get path of data from registry using provided ID
        path: Path = registry.get_path(identifier)
        # Import data from path
        return importer.import_data(path)

    def export_raw_feature_list(self, raw_feature_list: List[RawFeatureSetEntry]):
        data_type = type(RawFeatureSetEntry)
        # Export features
        exporter: Exporter = self.export_manager.get_provider(data_type)
        output_parent_dir: Path = self.output_dir_manager.get_provider(data_type)
        feature_id_to_output_path_map: Dict[str, Path] = {}
        feature_id_to_imu_data_id_map: Dict[Identifier, Identifier] = {}
        for feature in raw_feature_list:
            feature_id = feature.metadata.raw_feature_identifier
            imu_data_id = feature.metadata.imu_data_identifier
            output_path = exporter.export_data(output_parent_dir, feature)
            feature_id_to_output_path_map[feature_id.value] = output_path
            feature_id_to_imu_data_id_map[feature_id.value] = imu_data_id.value
        # Update raw feature registry
        # Update Raw feature -> IMU Data mapping
        pass

    def export_aggregate_feature_list(self):
        # Export features
        # Update aggregate feature registry
        # Update aggregate feature -> raw feature mapping
        pass

    def _update_registry(
        self,
        importer: Importer,
        exporter: Exporter,
        source_registry_path: Path,
        new_id_to_path_map: Dict[str, Path],
        data_type: Type[Identifier],
    ):
        source_registry: Registry = importer.import_data(source_registry_path)
        source_id_to_path_map: Dict[str, Path] = source_registry.registry
        if not self.__any_new_ids_in_source(source_id_to_path_map, new_id_to_path_map):
            new_mapping: Dict[str, Path] = self.__construct_mapping(
                source_id_to_path_map, new_id_to_path_map
            )
            new_registry = Registry(new_mapping, data_type, source_registry_path)
            exporter.export_data()
        else:
            raise ValueError(f"Attempting to add existing elements mapping")

    def _update_mapping(
        mapping: Mapping, source_id_to_target_id_map: Dict[Identifier, Identifier]
    ):
        pass

    def __any_new_ids_in_source(
        self, source_mapping: Dict[S, T], current_mapping: Dict[S, T]
    ) -> bool:
        return any(
            [new_id in source_mapping.keys() for new_id in current_mapping.keys()]
        )

    def __construct_new_mapping(
        self, source_mapping: Dict[S, T], current_mapping: Dict[S, T]
    ) -> Dict[S, T]:
        new_mapping = {k: v for k, v in source_mapping.items()}
        for k, v in current_mapping.items():
            new_mapping[k] = v
        return new_mapping

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
