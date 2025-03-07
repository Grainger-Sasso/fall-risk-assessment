from typing import Dict, Type

from src.data_io.import_export.exporters.exporter import Exporter
from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.identifiers.identifier import Identifier


class ExportManager(DataAccessManager[Exporter]):
    """Manages access to exporters"""

    def __init__(self, providers: Dict[Type[Identifier], Exporter]):
        super().__init__(providers)
