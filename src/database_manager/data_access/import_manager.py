from typing import Dict, Type

from src.data_io.import_export.importers.importer import Importer
from src.identifiers.identifier import Identifier


class ImportManager:
    """
    Import manager responsible for importing IMU data, User data, features, and instrument specifications
    """

    def __init__(self, importers: Dict[Type[Identifier], Importer]):
        self._importers: Dict[Type[Identifier], Importer] = importers

    @property
    def importers(self):
        return self._importers

    def get_importer(self, data_type: Type[Identifier]):
        if data_type not in self.importers:
            raise KeyError(f"Unable to resolve importer from ID: {data_type}")
        return self.importers[data_type]
