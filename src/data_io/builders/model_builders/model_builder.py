from abc import ABC, abstractmethod
from typing import Any

from src.data_io.formats.file_format import FileFormat


class ModelBuilder(ABC):
    version: str  # Model builder version

    def __init__(self):
        self.version = self.__class__.version

    @abstractmethod
    def build(self, input_file: FileFormat, **kwargs) -> Any:
        """Converts input file to data model

        Args:
            file (FileFormat): input file data

        Returns:
            Any: _description_
        """
        pass
