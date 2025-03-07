from typing import Dict, Type

from src.data_io.import_export.importers.importer import Importer
from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.identifiers.identifier import Identifier

class ImportManager(DataAccessManager[Importer]):
    """Manages access to importers"""

    def __init__(self, providers: Dict[Type[Identifier], Importer]):
        super().__init__(providers)
