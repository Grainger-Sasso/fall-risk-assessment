from typing import Dict, Type

from src.data_io.import_export.exporters.exporter import Exporter
from src.identifiers.identifier import Identifier


class ExportManager:
    """
    Export manager responsible for exporting IMU data and features
    """

    def __init__(self, exporters: Dict[Type[Identifier], Exporter]):
        self._exporters: Dict[Type[Identifier], Exporter] = exporters

    @property
    def exporters(self):
        return self._exporters

    def get_exporter(self, data_type: Type[Identifier]):
        if data_type not in self.exporters:
            raise KeyError(f"Unable to resolve exporter from ID: {data_type}")
        return self.exporters[data_type]
