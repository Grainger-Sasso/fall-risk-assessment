from typing import Dict, Type

from src.identifiers.identifier import Identifier


class Mapping:
    """Class for identifier mapping"""

    def __init__(
        self,
        map: Dict[str, str],
        source_id_type: Type[Identifier],
        target_id_type: Type[Identifier],
    ):
        self._map: Dict[str, str] = map
        self._source_id_type: Type[Identifier] = source_id_type
        self._target_id_type: Type[Identifier] = target_id_type

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
