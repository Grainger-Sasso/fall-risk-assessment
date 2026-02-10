from pathlib import Path
from typing import Dict, Tuple, Type

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.exporters.exporter import Exporter
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
from src.data_io.import_export.importers.dataset.dataset_importer import DatasetImporter
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
from src.data_model.dataset.dataset import Dataset
from src.gait_features.feature_extraction.gait_feature_extractor import (
    GaitFeatureExtractor,
    GaitResults,
)
from src.gait_features.feature_processing.gait_feature_processor import (
    GaitFeatureProcessor,
)
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


class FeatureExtractionPipeline:

    def __init__(
        self,
        registry_paths: Dict[Type[Identifier], Path],
        mapping_paths: Dict[Tuple[Type[Identifier]], Path],
        output_dir_paths: Dict[Type[Identifier], Path],
    ):
        # Assumes registry paths are to subdirectories containing existing registry files
        self.registry_paths: Dict[Type[Identifier], Path] = registry_paths
        # Assumes keys are (source ID type, target ID type); assumes mapping paths are to subdirectories containing existing mapping files
        self.mapping_paths: Dict[Type[Identifier], Path] = mapping_paths
        # Assumes output dir paths are to parent directories for data output
        self.output_dir_paths: Dict[Type[Identifier], Path] = output_dir_paths
        self.registry_importer = RegistryImporter()
        self.mapping_importer = MappingImporter()
        self.db_manager: DatabaseManager = self._setup_db_manager()
        self.gait_feature_extractor = GaitFeatureExtractor()
        self.gait_feature_processor = GaitFeatureProcessor()

    def run(self, dataset_parent_dir: Path, dataset_name: str):
        dataset_importer = DatasetImporter()
        dataset: Dataset = dataset_importer.import_data(
            dataset_parent_dir, dataset_name
        )
        for entry in dataset.entries:
            imu_data_id: IMUDataIdentifier = entry.imu_data_id
            user_data_id: UserIdentifier = entry.user_data_id
            imu_data: IMUData = self.db_manager.import_data([imu_data_id])[0]
            user_data: UserData = self.db_manager.import_data([user_data_id])[0]
            gait_res: GaitResults = self.gait_feature_extractor.extract_gait_features(
                imu_data, user_data
            )
            raw_features, agg_features = self.gait_feature_processor.process_features(
                gait_res, imu_data, user_data
            )
            self.db_manager.export_data([raw_features])
            self.db_manager.export_data([agg_features])

    def _setup_db_manager(self) -> DatabaseManager:
        # Setup registry manager
        registries: Dict[Type[Identifier], Registry] = {}
        for id_type, registry_path in self.registry_paths.items():
            registries[id_type] = self.registry_importer.import_data(
                registry_path, id_type
            )
        registry_manager = RegistryManager(registries)
        # Setup mapping manager
        mappings: Dict[Type[Identifier], Mapping] = {}
        for (
            source_id_type,
            target_id_type,
        ), mapping_path in self.mapping_paths.items():
            mappings[source_id_type] = self.mapping_importer.import_data(
                mapping_path, source_id_type, target_id_type
            )
        mapping_manager = MappingManager(mappings)
        # Setup output dir manager
        output_dir_manager = OutputDirectoryManager(self.output_dir_paths)
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
        # Setup database manager
        return DatabaseManager(
            registry_manager,
            mapping_manager,
            import_manager,
            export_manager,
            output_dir_manager,
        )
