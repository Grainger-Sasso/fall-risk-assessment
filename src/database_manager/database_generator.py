import os
from pathlib import Path
from typing import Dict, Optional, Type

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.exporters.feature.feature_file_exporter import (
    FeatureFileExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataImporter,
)
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataImporter,
)
from src.data_io.import_export.importers.features.feature_importer import (
    FeatureImporter,
)
from src.data_io.import_export.exporters.instrument_specifications.instrument_specification_file_exporter import (
    InstrumentSpecificationFileExporter,
)
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_validator import DatabaseValidator
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


class DatabaseGenerator:
    def __init__(
        self,
    ) -> None:
        self.validator = DatabaseValidator()

    def generate_database(
        self,
        output_dir_paths: Optional[Dict[Type, Path]] = None,
        sqlite_db_path: Optional[Path] = None,
        validate=True,
    ) -> DatabaseManager:
        if not output_dir_paths:
            raise ValueError("output_dir_paths is required")
        output_root = Path(
            os.path.commonpath([str(path.resolve()) for path in output_dir_paths.values()])
        )
        db_path = sqlite_db_path or (output_root / "index.db")
        sqlite_store = SQLiteStore(db_path)
        metadata_repository = MetadataRepository(sqlite_store)

        io_router = DomainIORouter(
            imu_importer=IMUDataImporter(),
            user_importer=UserDataImporter(),
            feature_importer=FeatureImporter(),
            instrument_spec_importer=InstrumentSpecificationImporter(),
            imu_exporter=IMUDataFileExporter(),
            feature_exporter=FeatureFileExporter(),
            instrument_spec_exporter=InstrumentSpecificationFileExporter(),
            imu_output_dir=output_dir_paths[IMUDataIdentifier],
            feature_output_dir=output_dir_paths[FeatureIdentifier],
            instrument_spec_output_dir=output_dir_paths.get(
                InstrumentSpecificationIdentifier
            ),
        )
        db_manager = DatabaseManager(
            metadata_repository=metadata_repository,
            io_router=io_router,
        )
        if validate:
            self.validator.validate_imu_data(db_manager)
            self.validator.validate_feature_data(db_manager)
            self.validator.validate_instrument_spec_data(db_manager)
        # Setup database manager
        return db_manager
