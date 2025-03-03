from pathlib import Path
from typing import Generic, TypeVar

from src.data_io.import_export.importers.importer import Importer

T = TypeVar("T")  # Type of data model object


class DataLoader(Generic[T]):
    """Loads data model objects using appropriate importers"""

    def __init__(self, importer: Importer[T]):
        self._importer = importer

    def load(self, path: Path) -> T:
        """Load data from a file path

        Args:
            path: Path to the data file

        Returns:
            Loaded data model object

        Raises:
            FileNotFoundError: If file doesn't exist
            ImportError: If import fails
        """
        if not path.exists():
            raise FileNotFoundError(f"No file found at path: {path}")

        try:
            return self._importer.import_data(path.parent)
        except Exception as e:
            raise ImportError(f"Failed to import data: {str(e)}")
