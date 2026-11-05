from pathlib import Path

from src.database_manager.database_generator import DatabaseGenerator
from src.data_visualization.suite.services.visualization_data_service import (
    VisualizationDataService,
)


def build_visualization_data_service(
    sqlite_db_path: Path, validate: bool = True
) -> VisualizationDataService:
    """Build a data service for visualization workflows over an existing index DB."""
    db_manager = DatabaseGenerator().generate_database(sqlite_db_path, validate=validate)
    return VisualizationDataService(db_manager=db_manager)
