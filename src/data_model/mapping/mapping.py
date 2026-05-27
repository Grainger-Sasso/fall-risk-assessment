from pathlib import Path
from typing import Dict, Type

from src.identifiers.identifier import Identifier


class Mapping:
    """Class for identifier mapping"""

    def __init__(
        self,
        map: Dict[str, str],
        source_id_type: Type[Identifier],
        target_id_type: Type[Identifier],
        subdir_path: Path,
    ):
        self._map: Dict[str, str] = map
        self._source_id_type: Type[Identifier] = source_id_type
        self._target_id_type: Type[Identifier] = target_id_type
        # Path to subdir containing mapping
        self._path: Path = subdir_path

    @property
    def map(self) -> Dict[str, str]:
        """Get the mapping dictionary

        Returns:
            Dictionary mapping source to target identifiers
        """
        return self._map

    @map.setter
    def map(self, map: Dict[str, str]) -> None:
        """Set the mapping dictionary

        Args:
            map: Dictionary mapping source to target identifiers
        """
        self._map = map

    @property
    def source_id_type(self) -> Type[Identifier]:
        return self._source_id_type

    @property
    def target_id_type(self) -> Type[Identifier]:
        return self._target_id_type

    @property
    def path(self) -> Path:
        return self._path

    def get_target_id_from_source_id(self, source_id: Identifier) -> Identifier:
        self._validate_source_id_type(type(source_id))
        self._validate_id_in_map(source_id)
        return self.target_id_type(self.map[source_id.value])

    def add_entry(self, source_id: Identifier, target_id: Identifier):
        self._validate_source_id_type(type(source_id))
        self._validate_target_id_type(type(target_id))
        if source_id.value in self.map.keys():
            raise ValueError(f"Entry already exists in registry for {source_id.value}")
        self.map[source_id.value] = target_id.value

    def update_entry(self, source_id: Identifier, target_id: Identifier):
        self._validate_source_id_type(type(source_id))
        self._validate_id_in_map(source_id)
        self._validate_target_id_type(type(target_id))
        self.map[source_id.value] = target_id.value

    def _validate_source_id_type(self, source_id_type: Type[Identifier]):
        if source_id_type != self._source_id_type:
            raise ValueError(
                f"Invalid source identifier type: expected type - {self._source_id_type}, recieved type - {source_id_type}"
            )

    def _validate_target_id_type(self, target_id_type: Type[Identifier]):
        if target_id_type != self._target_id_type:
            raise ValueError(
                f"Invalid target identifier type: expected type - {self._target_id_type}, recieved type - {target_id_type}"
            )

    def _validate_id_in_map(self, identifier: Identifier):
        if identifier.value not in self.map.keys():
            raise KeyError(f"Unable to resolve path from ID: {identifier}")
