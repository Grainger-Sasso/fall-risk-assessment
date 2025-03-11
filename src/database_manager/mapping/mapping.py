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

    def get_target_id(self, source_id: Identifier) -> Identifier:
        if source_id.value not in self.map.keys():
            raise KeyError(f"Unable to resolve target ID of source ID: {source_id}")
        return self.target_id_type(self.map[source_id.value])
