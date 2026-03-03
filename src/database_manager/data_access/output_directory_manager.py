from pathlib import Path
from typing import Dict, Type

from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.identifiers.identifier import Identifier


class OutputDirectoryManager(DataAccessManager[Path]):
    """Manages access to output directory paths"""

    def __init__(self, providers: Dict[Type[Identifier], Path]):
        super().__init__(providers)
        self.validate_output_directories()

    def validate_output_directories(self) -> None:
        """Validates all output directory paths across all providers."""
        for id_type, path in self.providers.items():
            if path is None:
                raise ValueError(
                    f"Output directory for type {id_type} is None"
                )
            if not isinstance(path, Path):
                raise ValueError(
                    f"Output directory for type {id_type} is not a Path: {type(path)}"
                )
            if not path.exists():
                raise ValueError(
                    f"Output directory for type {id_type} does not exist: {path}"
                )
            if not path.is_dir():
                raise ValueError(
                    f"Output directory for type {id_type} is not a directory: {path}"
                )
