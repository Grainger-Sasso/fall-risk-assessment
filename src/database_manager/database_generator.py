from pathlib import Path
from typing import Optional

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
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_validator import DatabaseValidator
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore


class DatabaseGenerator:
    """Open an existing SQLite index as a ``DatabaseManager`` and validate it.

    The index itself is built beforehand by ``build_sql_index``. The feature
    output directory defaults to the index's parent directory.
    """

    def __init__(self) -> None:
        self.validator = DatabaseValidator()

    def generate_database(
        self,
        sqlite_db_path: Path,
        feature_output_dir: Optional[Path] = None,
        validate: bool = True,
    ) -> DatabaseManager:
        sqlite_db_path = Path(sqlite_db_path).resolve()
        if not sqlite_db_path.is_file():
            raise FileNotFoundError(
                f"SQLite index not found: {sqlite_db_path}. Build it with build_sql_index first."
            )
        default_output_dir = sqlite_db_path.parent

        io_router = DomainIORouter(
            imu_importer=IMUDataImporter(),
            user_importer=UserDataImporter(),
            feature_importer=FeatureImporter(),
            instrument_spec_importer=InstrumentSpecificationImporter(),
            feature_exporter=FeatureFileExporter(),
            feature_output_dir=(feature_output_dir or default_output_dir).resolve(),
        )
        db_manager = DatabaseManager(
            metadata_repository=MetadataRepository(SQLiteStore(sqlite_db_path)),
            io_router=io_router,
        )
        if validate:
            self.validator.validate_database(db_manager)
        return db_manager
