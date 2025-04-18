from pathlib import Path
from typing import Dict, List, Type, TypeVar

from src.data_io.import_export.exporters.database_exporter import DatabaseExporter
from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.exporters.mapping.mapping_exporter import MappingExporter
from src.data_io.import_export.exporters.registry.registry_exporter import (
    RegistryExporter,
)
from src.data_io.import_export.importers.importer import Importer
from src.data_model.assessment_data import AssessmentData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
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
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.identifier import Identifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier

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

    ### I/O Methods ###
    def import_data(self, identifier: Identifier) -> T:
        data_type: Type[Identifier] = type(identifier)
        # Get corresponding registry and importer
        registry: Registry = self.registry_manager.get_provider(data_type)
        importer: Importer = self.import_manager.get_provider(data_type)
        # Get path of data from registry using provided ID
        path: Path = registry.get_path_from_id(identifier)
        # Import data from path
        return importer.import_data(path)

    def export_data(self, assessment_data_list: List[AssessmentData]) -> None:
        if len(set(type(data.get_data_id()) for data in assessment_data_list)) != 1:
            raise ValueError(
                "All elements for export must share common data type (source identifier type)"
            )
        # Reference data_type from type(assessment_data.get_data_id())
        source_data_type = type(assessment_data_list[0].get_data_id())
        # Get exporter, output dir, registry, mapping (from type of assessment_data)
        output_dir: Path = self.output_dir_manager.get_provider(source_data_type)
        exporter: Exporter = self.export_manager.get_provider(source_data_type)
        registry: Registry = self.registry_manager.get_provider(source_data_type)
        mapping: Mapping = self.mapping_manager.get_provider(source_data_type)
        # For every item in assessment data
        for asessment_data in assessment_data_list:
            source_data_id: Identifier = asessment_data.get_data_id()
            target_data_id: Identifier = asessment_data.get_associated_data_id()
            # Export data with exporter by passing output dir and data object
            output_path: Path = exporter.export_data(output_dir, asessment_data)
            # Call update registry method on registry
            registry.add_entry(source_data_id, output_path)
            # Call update mapping on mapping
            if target_data_id:
                mapping.add_entry(source_data_id, target_data_id)
        # Export updated registry
        registry_exporter: RegistryExporter = RegistryExporter()
        registry_exporter.export_data(registry.path, registry)
        # Export updated mapping
        mapping_exporter: MappingExporter = MappingExporter()
        mapping_exporter.export_data(mapping.path, mapping)

    def export_raw_feature_list(self, raw_feature_list: List[RawFeatureSetEntry]):
        if not raw_feature_list:
            raise ValueError("Feature list cannot be empty")

        data_type = RawFeatureIdentifier
        # Get output directory
        output_parent_dir: Path = self.output_dir_manager.get_provider(data_type)
        if output_parent_dir is None:
            raise ValueError("No output directory configured")

        # Export features
        exporter: Exporter = self.export_manager.get_provider(data_type)
        feature_id_to_output_path_map: Dict[str, Path] = {}
        feature_id_to_imu_data_id_map: Dict[RawFeatureIdentifier, IMUDataIdentifier] = (
            {}
        )

        try:
            for feature in raw_feature_list:
                feature_id = feature.metadata.raw_feature_identifier
                imu_data_id = feature.metadata.imu_data_identifier
                output_path = exporter.export_data(output_parent_dir, feature)
                feature_id_to_output_path_map[feature_id.value] = output_path
                feature_id_to_imu_data_id_map[feature_id.value] = imu_data_id.value

            # Update raw feature registry
            raw_feature_registry: Registry = self.registry_manager.get_provider(
                data_type
            )
            registry_exporter: RegistryExporter = RegistryExporter()
            self._update_registry(
                registry_exporter,
                raw_feature_registry,
                feature_id_to_output_path_map,
                data_type,
            )

            # Update Raw feature -> IMU Data mapping
            raw_feature_mapping: Mapping = self.mapping_manager.get_provider(data_type)
            mapping_exporter: MappingExporter = MappingExporter()
            self._update_mapping(
                mapping_exporter,
                raw_feature_mapping,
                feature_id_to_imu_data_id_map,
                data_type,
                IMUDataIdentifier,
            )
        except Exception as e:
            # Re-raise any exceptions that occur during export
            raise Exception(f"Export failed: {str(e)}")

    def export_aggregate_feature_list(
        self, agg_feature_list: List[AggregateFeatureSetEntry]
    ):
        if not agg_feature_list:
            raise ValueError("Feature list cannot be empty")

        data_type = AggregateFeatureIdentifier
        # Get output directory
        output_parent_dir: Path = self.output_dir_manager.get_provider(data_type)
        if output_parent_dir is None:
            raise ValueError("No output directory configured")

        # Export features
        exporter: Exporter = self.export_manager.get_provider(data_type)
        feature_id_to_output_path_map: Dict[str, Path] = {}
        agg_id_to_raw_id_map: Dict[AggregateFeatureIdentifier, RawFeatureIdentifier] = (
            {}
        )

        try:
            for feature in agg_feature_list:
                agg_feature_id = feature.metadata.aggregate_feature_identifier
                raw_feature_id = feature.metadata.raw_feature_identifier
                output_path = exporter.export_data(output_parent_dir, feature)
                feature_id_to_output_path_map[agg_feature_id.value] = output_path
                agg_id_to_raw_id_map[agg_feature_id.value] = raw_feature_id.value

            # Update agg feature registry
            agg_feature_registry: Registry = self.registry_manager.get_provider(
                data_type
            )
            registry_exporter: RegistryExporter = RegistryExporter()
            self._update_registry(
                registry_exporter,
                agg_feature_registry,
                feature_id_to_output_path_map,
                data_type,
            )

            # Update agg feature -> raw feature mapping
            agg_feature_mapping: Mapping = self.mapping_manager.get_provider(data_type)
            mapping_exporter: MappingExporter = MappingExporter()
            self._update_mapping(
                mapping_exporter,
                agg_feature_mapping,
                agg_id_to_raw_id_map,
                data_type,
                RawFeatureIdentifier,
            )
        except Exception as e:
            # Re-raise any exceptions that occur during export
            raise Exception(f"Export failed: {str(e)}")

    def _update_registry(
        self,
        exporter: DatabaseExporter,
        source_registry: Registry,
        new_id_to_path_map: Dict[str, Path],
        data_type: Type[Identifier],
    ):
        source_id_to_path_map: Dict[str, Path] = source_registry.registry
        source_registry_subdir_path: Path = source_registry.path
        if not self._any_new_ids_in_source(source_id_to_path_map, new_id_to_path_map):
            new_mapping: Dict[str, Path] = self._construct_new_mapping(
                source_id_to_path_map, new_id_to_path_map
            )
            new_registry = Registry(new_mapping, data_type, source_registry_subdir_path)
            exporter.export_data(source_registry_subdir_path, new_registry)
            # Update registry manager
            self.registry_manager.set_provider(data_type, new_registry)
        else:
            raise ValueError(f"Attempting to add existing elements to registry")

    def _update_mapping(
        self,
        exporter: DatabaseExporter,
        source_mapping: Mapping,
        new_source_id_to_target_id_map: Dict[str, str],
        data_type: Type[Identifier],
        target_id_type: Type[Identifier],
    ):
        source_id_to_target_id_map: Dict[str, str] = source_mapping.map
        source_mapping_subdir_path: Path = source_mapping.path
        if not self._any_new_ids_in_source(
            source_id_to_target_id_map, new_source_id_to_target_id_map
        ):
            new_mapping: Dict[str, str] = self._construct_new_mapping(
                source_id_to_target_id_map, new_source_id_to_target_id_map
            )
            new_dataset_mapping = Mapping(
                new_mapping, data_type, target_id_type, source_mapping_subdir_path
            )
            exporter.export_data(source_mapping_subdir_path, new_dataset_mapping)
            # Update mapping manager
            self.mapping_manager.set_provider(data_type, new_dataset_mapping)
        else:
            raise ValueError(f"Attempting to add existing elements to mapping")

    def _any_new_ids_in_source(
        self, source_mapping: Dict[S, T], current_mapping: Dict[S, T]
    ) -> bool:
        return any(
            [new_id in source_mapping.keys() for new_id in current_mapping.keys()]
        )

    def _construct_new_mapping(
        self, source_mapping: Dict[S, T], current_mapping: Dict[S, T]
    ) -> Dict[S, T]:
        new_mapping = {k: v for k, v in source_mapping.items()}
        for k, v in current_mapping.items():
            new_mapping[k] = v
        return new_mapping

    ###

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
