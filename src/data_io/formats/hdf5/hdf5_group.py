from dataclasses import dataclass
from typing import Any, List, Dict, Union

from data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.file_format import FileFormat


@dataclass
class HDF5Group(FileFormat):
    name: str
    # List of unique items (by name)
    items: List[Union["HDF5Group", HDF5Dataset]]
    attributes: Dict[str, Any]

    def get_item_by_name(self, name: str) -> Union["HDF5Group", HDF5Dataset]:
        """Gets item by name assuming unique names

        Args:
            name (str): name of item

        Returns:
            _type_: _description_
        """
        for item in self.items:
            if item.name == name:
                return item
        raise ValueError(f"No item with name '{name}' found.")
