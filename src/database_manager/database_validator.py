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


class DatabaseValidator:
    def __init__(self):
        pass
    
    def validate_imu_data(self, db_manager: DatabaseManager):
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
                imu_data: IMUData = db_manager.import_data([IMUDataIdentifier(imu_id)])[0]
                imu_id_from_data: Identifier = imu_data.get_data_id()
                user_id: Identifier = imu_data.get_associated_data_id()
                print(f'Validating data for user: {user_id.value}')
                # Read in  imu data and check ID for imu and user
                if imu_id != imu_id_from_data.value:
                    raise ValueError(
                        f"IMU data ID in registry -{imu_id.value}- does not match ID in file -{imu_id_from_data.value}-"
                    )
                if user_id.value not in user_data_registry.registry.keys():
                    raise ValueError(
                        f"For IMU ID -{imu_id.value}-: User ID in not present in user data registry -{user_id.value}-"
                    )
                if user_id.value != user_id_from_mapping.value:
                    raise ValueError(
                        f"For IMU ID -{imu_id.value}-: User ID in mapping -{user_id_from_mapping.value}- does not match ID in file -{user_id.value}-"
                    )
                # Read in the user data found from IMU data and check ID
                user_data: UserData = db_manager.import_data([user_id])[0]
                user_id_from_data: Identifier = user_data.get_data_id()
                if user_id_from_data.value != user_id.value:
                    raise ValueError(
                        f"User data identifier from imu data -{user_id.value}- does not match ID in file -{user_id_from_data.value}-"
                    )
            # Read in the user data present there
            except Exception as e:
                raise Exception(e)
        return True

    def validate_raw_features(self, db_manager: DatabaseManager):
        # For every raw feature file present in the raw feature registry
        raw_feature_registry: Registry = db_manager.registry_manager.get_provider(
            RawFeatureIdentifier
        )
        imu_data_registry: Registry = db_manager.registry_manager.get_provider(
            IMUDataIdentifier
        )
        raw_feature_to_imu_map: Mapping = db_manager.mapping_manager.get_provider(
            RawFeatureIdentifier
        )
        for raw_feature_id in raw_feature_registry.registry.keys():
            raw_feature_id: str
            imu_id_from_mapping: Identifier = raw_feature_to_imu_map.get_target_id_from_source_id(
                RawFeatureIdentifier(raw_feature_id)
            )
            try:
                raw_feature_data: RawFeatureSetEntry = db_manager.import_data(
                    [RawFeatureIdentifier(raw_feature_id)]
                )[0]
                raw_feature_id_from_data: Identifier = raw_feature_data.get_data_id()
                imu_id: Identifier = raw_feature_data.get_associated_data_id()
                print(f"Validating raw features for IMU data: {imu_id.value}")
                if raw_feature_id != raw_feature_id_from_data.value:
                    raise ValueError(
                        f"Raw feature ID in registry -{raw_feature_id}- does not match ID in file -{raw_feature_id_from_data.value}-"
                    )
                if imu_id.value not in imu_data_registry.registry.keys():
                    raise ValueError(
                        f"For raw feature ID -{raw_feature_id}-: IMU ID not present in IMU data registry -{imu_id.value}-"
                    )
                if imu_id.value != imu_id_from_mapping.value:
                    raise ValueError(
                        f"For raw feature ID -{raw_feature_id}-: IMU ID in mapping -{imu_id_from_mapping.value}- does not match ID in file -{imu_id.value}-"
                    )
                imu_data: IMUData = db_manager.import_data([imu_id])[0]
                imu_id_from_data: Identifier = imu_data.get_data_id()
                if imu_id_from_data.value != imu_id.value:
                    raise ValueError(
                        f"IMU data identifier from raw feature -{imu_id.value}- does not match ID in file -{imu_id_from_data.value}-"
                    )
            except Exception as e:
                raise Exception(e)
        return True

    def validate_aggregate_features(self, db_manager: DatabaseManager):
        # For every aggregate feature file present in the aggregate feature registry
        aggregate_feature_registry: Registry = db_manager.registry_manager.get_provider(
            AggregateFeatureIdentifier
        )
        raw_feature_registry: Registry = db_manager.registry_manager.get_provider(
            RawFeatureIdentifier
        )
        aggregate_feature_to_raw_map: Mapping = db_manager.mapping_manager.get_provider(
            AggregateFeatureIdentifier
        )
        for agg_feature_id in aggregate_feature_registry.registry.keys():
            agg_feature_id: str
            raw_feature_id_from_mapping: Identifier = (
                aggregate_feature_to_raw_map.get_target_id_from_source_id(
                    AggregateFeatureIdentifier(agg_feature_id)
                )
            )
            try:
                agg_feature_data: AggregateFeatureSetEntry = db_manager.import_data(
                    [AggregateFeatureIdentifier(agg_feature_id)]
                )[0]
                agg_feature_id_from_data: Identifier = agg_feature_data.get_data_id()
                raw_feature_id: Identifier = agg_feature_data.get_associated_data_id()
                print(f"Validating aggregate features for raw feature: {raw_feature_id.value}")
                if agg_feature_id != agg_feature_id_from_data.value:
                    raise ValueError(
                        f"Aggregate feature ID in registry -{agg_feature_id}- does not match ID in file -{agg_feature_id_from_data.value}-"
                    )
                if raw_feature_id.value not in raw_feature_registry.registry.keys():
                    raise ValueError(
                        f"For aggregate feature ID -{agg_feature_id}-: Raw feature ID not present in raw feature registry -{raw_feature_id.value}-"
                    )
                if raw_feature_id.value != raw_feature_id_from_mapping.value:
                    raise ValueError(
                        f"For aggregate feature ID -{agg_feature_id}-: Raw feature ID in mapping -{raw_feature_id_from_mapping.value}- does not match ID in file -{raw_feature_id.value}-"
                    )
                raw_feature_data: RawFeatureSetEntry = db_manager.import_data(
                    [raw_feature_id]
                )[0]
                raw_feature_id_from_data: Identifier = raw_feature_data.get_data_id()
                if raw_feature_id_from_data.value != raw_feature_id.value:
                    raise ValueError(
                        f"Raw feature identifier from aggregate feature -{raw_feature_id.value}- does not match ID in file -{raw_feature_id_from_data.value}-"
                    )
            except Exception as e:
                raise Exception(e)
        return True