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
    def import_data(self, identifier_list: List[Identifier]) -> List[T]:
        if len(set(type(identifier) for identifier in identifier_list)) != 1:
            raise ValueError(
                "All elements for import must share common data type (identifier type)"
            )
        data_type: Type[Identifier] = type(identifier_list[0])
        # Get corresponding registry and importer
        registry: Registry = self.registry_manager.get_provider(data_type)
        importer: Importer = self.import_manager.get_provider(data_type)
        data: List[T] = []
        for identifier in identifier_list:
            # Get path of data from registry using provided ID
            path: Path = registry.get_path_from_id(identifier)
            # Import data from path
            data.append(importer.import_data(path))
        return data

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
