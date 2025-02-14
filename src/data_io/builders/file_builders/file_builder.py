from abc import ABC, abstractmethod
from typing import Any

from src.data_io.formats.file_format import FileFormat


class FileBuilder(ABC):
    """Interface for file builders"""

    version: str

    def __init__(self):
        self.version = self.__class__.version

    @abstractmethod
    def build(self, data: Any) -> FileFormat:
        """Build file format from input data (model object)

        Args:
            data (Any): Model object to be written to file

        Returns:
            FileFormat: Output file
        """
        pass
