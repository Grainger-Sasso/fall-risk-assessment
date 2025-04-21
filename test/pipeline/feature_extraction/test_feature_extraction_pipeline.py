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
from src.data_processing.feature_extraction.gait_feature_extractor import (
    GaitFeatureExtractor,
    GaitResults,
)
from src.data_processing.feature_processing.gait_feature_processor import (
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
from src.pipeline.feature_extraction.feature_extraction_pipeline import (
    FeatureExtractionPipeline,
)


def main():
    registry_paths = {
        IMUDataIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/registries/imu_data_registry"
        ),
        UserIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/registries/user_data_registry"
        ),
        RawFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/registries/raw_feature_registry"
        ),
        AggregateFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/registries/agg_feature_registry"
        ),
    }
    mapping_paths = {
        (IMUDataIdentifier, UserIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/mappings/imu_data_mapping"
        ),
        (RawFeatureIdentifier, IMUDataIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/mappings/raw_feature_mapping"
        ),
        (AggregateFeatureIdentifier, RawFeatureIdentifier): Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/mappings/agg_feature_mapping"
        ),
    }
    output_dir_paths = {
        RawFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/output/raw_features"
        ),
        AggregateFeatureIdentifier: Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/output/agg_features"
        ),
    }
    pipeline = FeatureExtractionPipeline(
        registry_paths, mapping_paths, output_dir_paths
    )
    pipeline.run(
        Path(
            "/Users/graingersasso/Desktop/fafra_testing/feature_extraction/input/dataset"
        ),
        "test_feature_ex_dataset",
    )
    


if __name__ == "__main__":
    main()
