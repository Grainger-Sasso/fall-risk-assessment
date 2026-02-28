from typing import Dict, Type

from src.database_manager.data_access.data_access_manager import DataAccessManager
from src.database_manager.mapping.mapping import Mapping
from src.identifiers.identifier import Identifier


class MappingManager(DataAccessManager[Mapping]):
    """Manages access to mappings"""

    def __init__(self, providers: Dict[Type[Identifier], Mapping]):
        super().__init__(providers)
        self.validate_mappings()

    def validate_mappings(self) -> None:
        """Validates all mapping entries across all providers."""
        for source_id_type, mapping in self.providers.items():
            if source_id_type != mapping.source_id_type:
                raise ValueError(
                    f"Mapping provider key {source_id_type} does not match "
                    f"mapping source_id_type {mapping.source_id_type}"
                )
            for source_id, target_id in mapping.map.items():
                if source_id is None or (isinstance(source_id, str) and source_id.strip() == ""):
                    raise ValueError(
                        f"Mapping of type {source_id_type} contains null or empty source id"
                    )
                if target_id is None or (isinstance(target_id, str) and target_id.strip() == ""):
                    raise ValueError(
                        f"Mapping of type {source_id_type} contains null or empty target id "
                        f"for source id '{source_id}'"
                    )
                try:
                    mapping.source_id_type(source_id)
                except ValueError as e:
                    raise ValueError(
                        f"Mapping of type {source_id_type} contains invalid source id "
                        f"'{source_id}': {e}"
                    ) from e
                try:
                    mapping.target_id_type(target_id)
                except ValueError as e:
                    raise ValueError(
                        f"Mapping of type {source_id_type} contains invalid target id "
                        f"'{target_id}' for source id '{source_id}': {e}"
                    ) from e
