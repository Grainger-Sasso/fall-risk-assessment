from typing import Dict


class Mapping:
    """Class for identifier mapping"""

    def __init__(self, map: Dict[str, str]):
        self._map: Dict[str, str] = map

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
