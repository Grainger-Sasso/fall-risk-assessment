from pathlib import Path
from typing import Dict, Type

from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier


class RegistryManager(DataAccessManager[Registry]):
    """Manages access to registries"""

    def __init__(self, providers: Dict[Type[Identifier], Registry]):
        super().__init__(providers)
        self.validate_registries()

    def validate_registries(self):
        for reg_type, registry in self.providers.items():
            for id, path in registry.registry.items():
                id:str
                path: Path
                if id is None or id == '':
                    raise ValueError(
                        f"Registry of type {str(reg_type)} contains null or empty id"
                    )
                if not path.exists():
                    raise ValueError(
                        f"Registry of type {str(reg_type)} contains path for id - {id} - that do not exist: {path}"
                    )
                if not path.is_dir():
                    raise ValueError(
                        f"Registry of type {str(reg_type)} contains path for id - {id} - that is not a directory: {path}"
                    )
                try:
                    if not any(p.is_file() for p in path.iterdir()):
                        raise ValueError(
                            f"Registry of type {str(reg_type)} contains empty directory for id - {id} - : {path}"
                        )
                except PermissionError:
                    raise PermissionError(
                        f"Registry of type {str(reg_type)} contains path for id - {id} - that is privileged: {path}"
                    )
