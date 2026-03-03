from pathlib import Path
from typing import Dict, Tuple, Type

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.exporters.feature.aggregate.aggregate_feature_file_exporter import (
    AggregateFeatureFileExporter,
)
from src.data_io.import_export.exporters.feature.raw.raw_feature_file_exporter import (
    RawFeatureFileExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataImporter,
)
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataImporter,
)
from src.data_io.import_export.importers.features.aggregate.aggregate_feature_importer import (
    AggregateFeatureImporter,
)
from src.data_io.import_export.importers.features.raw.raw_feature_importer import (
    RawFeatureImporter,
)
from src.data_io.import_export.importers.importer import Importer
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.data_io.import_export.importers.mapping.mapping_importer import MappingImporter
from src.data_io.import_export.importers.registry.registry_importer import (
    RegistryImporter,
)
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.database_manager.data_access.export_manager import ExportManager
from src.database_manager.data_access.import_manager import ImportManager
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.output_directory_manager import (
    OutputDirectoryManager,
)
from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.identifier import Identifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier


class DatabaseGenerator:
    def __init__(
        self,
    ) -> None:
        pass

    def generate_database(
        self,
        registry_paths: Dict[Type[Identifier], Path],
        mapping_paths: Dict[Tuple[Type[Identifier]], Path],
        output_dir_paths: Dict[Type[Identifier], Path],
        validate=True,
    ) -> DatabaseManager:

        registry_importer = RegistryImporter()
        mapping_importer = MappingImporter()
        # Setup registry manager
        registries: Dict[Type[Identifier], Registry] = {}
        for id_type, registry_path in registry_paths.items():
            registries[id_type] = registry_importer.import_data(registry_path, id_type)
        registry_manager = RegistryManager(registries)
        # Setup mapping manager
        mappings: Dict[Type[Identifier], Mapping] = {}
        for (
            source_id_type,
            target_id_type,
        ), mapping_path in mapping_paths.items():
            mappings[source_id_type] = mapping_importer.import_data(
                mapping_path, source_id_type, target_id_type
            )
        mapping_manager = MappingManager(mappings)
        # Setup output dir manager
        output_dir_manager = OutputDirectoryManager(output_dir_paths)
        # Setup import manager
        importers: Dict[Type[Identifier], Importer] = {
            IMUDataIdentifier: IMUDataImporter(),
            UserIdentifier: UserDataImporter(),
            RawFeatureIdentifier: RawFeatureImporter(),
            AggregateFeatureIdentifier: AggregateFeatureImporter(),
            InstrumentSpecificationIdentifier: InstrumentSpecificationImporter(),
        }
        import_manager = ImportManager(importers)
        # Setup export manager
        exporters: Dict[Type[Identifier], Importer] = {
            IMUDataIdentifier: IMUDataFileExporter(),
            RawFeatureIdentifier: RawFeatureFileExporter(),
            AggregateFeatureIdentifier: AggregateFeatureFileExporter(),
        }
        export_manager = ExportManager(exporters)
        db_manager = DatabaseManager(
            registry_manager,
            mapping_manager,
            import_manager,
            export_manager,
            output_dir_manager,
        )
        if validate:
            self._validate_database(db_manager)
        # Setup database manager
        return db_manager

    def _validate_database(db_manager: DatabaseManager):
        # For every IMU data file present in the imu data registry
        imu_data_registry: Registry = db_manager.registry_manager.get_provider(
            IMUDataIdentifier
        )
        user_data_registry: Registry = db_manager.registry_manager.get_provider(
            UserIdentifier
        )
        imu_to_user_map: Mapping = db_manager.mapping_manager.get_provider(
            IMUDataIdentifier
        )
        for imu_id in imu_data_registry.registry.keys():
            imu_id: str
            user_id_from_mapping: Identifier = (
                imu_to_user_map.get_target_id_from_source_id(IMUDataIdentifier(imu_id))
            )
            try:
                imu_data: IMUData = db_manager.import_data([IMUDataIdentifier(imu_id)])
                imu_id_from_data: Identifier = imu_data.get_data_id()
                user_id: Identifier = imu_data.get_associated_data_id()
                # Read in  imu data and check ID for imu and user
                if imu_id != imu_id_from_data.value:
                    raise ValueError(
                        f"IMU data ID in registry -{imu_id.value}- does not match ID in file -{imu_id_from_data.value}-"
                    )
                if user_id.value not in user_data_registry.keys():
                    raise ValueError(
                        f"For IMU ID -{imu_id.value}-: User ID in not present in user data registry -{user_id.value}-"
                    )
                if user_id.value != user_id_from_mapping.value:
                    raise ValueError(
                        f"For IMU ID -{imu_id.value}-: User ID in mapping -{user_id_from_mapping.value}- does not match ID in file -{user_id.value}-"
                    )
                # Read in the user data found from IMU data and check ID
                user_data: UserData = db_manager.import_data([user_id])
                user_id_from_data: Identifier = user_data.get_data_id()
                if user_id_from_data.value != user_id.value:
                    raise ValueError(
                        f"User data identifier from imu data -{user_id.value}- does not match ID in file -{user_id_from_data.value}-"
                    )
            # Read in the user data present there
            except Exception as e:
                raise Exception(e)
        return True
