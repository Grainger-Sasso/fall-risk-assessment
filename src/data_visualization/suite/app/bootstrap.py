from pathlib import Path

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.exporters.feature.feature_file_exporter import (
    FeatureFileExporter,
)
from src.data_io.import_export.exporters.instrument_specifications.instrument_specification_file_exporter import (
    InstrumentSpecificationFileExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import IMUDataImporter
from src.data_io.import_export.importers.data.user.user_data_importer import UserDataImporter
from src.data_io.import_export.importers.features.feature_importer import FeatureImporter
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.data_visualization.suite.services.visualization_data_service import (
    VisualizationDataService,
)


def build_visualization_data_service(sqlite_db_path: Path) -> VisualizationDataService:
    """Build a data service for visualization workflows over an existing index DB."""
    sqlite_db_path = sqlite_db_path.resolve()
    scratch_dir = sqlite_db_path.parent
    repository = MetadataRepository(SQLiteStore(sqlite_db_path))
    io_router = DomainIORouter(
        imu_importer=IMUDataImporter(),
        user_importer=UserDataImporter(),
        feature_importer=FeatureImporter(),
        instrument_spec_importer=InstrumentSpecificationImporter(),
        imu_exporter=IMUDataFileExporter(),
        feature_exporter=FeatureFileExporter(),
        instrument_spec_exporter=InstrumentSpecificationFileExporter(),
        imu_output_dir=scratch_dir,
        feature_output_dir=scratch_dir,
        instrument_spec_output_dir=scratch_dir,
    )
    db_manager = DatabaseManager(metadata_repository=repository, io_router=io_router)
    return VisualizationDataService(db_manager=db_manager)
