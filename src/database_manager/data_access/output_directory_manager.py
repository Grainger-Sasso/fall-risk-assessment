from pathlib import Path
from typing import Dict, Type

from src.data_io.import_export.exporters.exporter import Exporter
from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.identifiers.identifier import Identifier


class OutputDirectoryManager(DataAccessManager[Path]):
    """Manages access to output directory paths"""

    def __init__(self, providers: Dict[Type[Identifier], Path]):
        super().__init__(providers)
