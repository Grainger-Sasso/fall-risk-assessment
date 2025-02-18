from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from src.data_io.formats.file_format import FileFormat
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset


@dataclass
class HDF5Group(FileFormat):
    """HDF5 group containing datasets and attributes."""

    name: str
    items: List[Union["HDF5Group", HDF5Dataset]]
    attributes: Dict[str, Any]

    def get_item_by_name(self, name: str) -> Union["HDF5Group", HDF5Dataset]:
        """Gets item by name assuming unique names.

        Args:
            name (str): name of item

        Returns:
            Union[HDF5Group, HDF5Dataset]: The found item

        Raises:
            ValueError: If no item with the given name is found
        """
        for item in self.items:
            if item.name == name:
                return item
        raise ValueError(f"No item with name '{name}' found.")
