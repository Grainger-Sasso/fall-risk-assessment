from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple


class FileWriter(ABC):
    """
    Generic file writer interface
    """

    @abstractmethod
    def write(self, path: Path, data: Any, **kwargs) -> Tuple[bool, str]:
        """
        Writes data to a file.

        Args:
            path (Path): File path to write.
            data (DataType): Data to write. Expected to be a dictionary or list of dictionaries.
            **kwargs: Additional parameters for specific file formats (e.g., compression, delimiter).

        Returns:
            Tuple[bool, str]: (success flag, error message)
        """
        pass
