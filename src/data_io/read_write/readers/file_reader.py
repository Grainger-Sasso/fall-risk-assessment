from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class FileReader(ABC):
    """
    Generic file reader interface
    """

    @abstractmethod
    def read(self, path: Path, **kwargs) -> Any:
        """
        Generic read method for files
        Args:
            path (Path): file path to read
            **kwargs: Additional parameters for specific file formats (e.g., compression, delimiter).


        Returns:
            Any: Artifact from file
        """
        pass
