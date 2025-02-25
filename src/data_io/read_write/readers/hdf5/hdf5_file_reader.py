import heapq
from pathlib import Path
from typing import Tuple, Union


import h5py

from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.read_write.readers.file_reader import FileReader


class HDF5FileReader(FileReader):
    """
    HDF5 file reader.
    """

    def read(self, path: Path) -> HDF5Group:
        """Reads data from an HDF5 file and returns them Group format.

        Args:
            path (Path): The path to the HDF5 file.

        Returns:
            Group: A Group containing datasets, groups, and their attributes.
        """
        if path.suffix not in [".h5", ".hdf5"]:
            raise ValueError(
                f"Expected an HDF5 file (.h5 or .hdf5), but got {path.suffix}."
            )

        if not path.exists():
            raise FileNotFoundError(f"The file at {path} does not exist.")

        with h5py.File(path, "r") as file:
            parent_name_and_group = [i for i in file.items()][0]

            def recursively_load_hdf5(
                name_and_group: Tuple[str, Union[h5py.Group, h5py.Dataset]]
            ) -> HDF5Group:
                """Recursively reads a Group object to an HDF5 group.

                Args:
                    group (h5py.Group): The Group object to read.
                """
                name = name_and_group[0]
                group = name_and_group[1]
                # Init items
                output_group_items = []
                # For all items in group items
                for key, item in group.items():
                    # If item is dataset
                    if isinstance(item, h5py.Dataset):
                        # Create dataset and add to output
                        output_dataset = HDF5Dataset(
                            name=key,
                            data=item[()],
                            attributes={key: val for key, val in item.attrs.items()},
                        )
                        output_group_items.append(output_dataset)
                    # Else is a group
                    elif isinstance(item, h5py.Group):
                        # Make recursive call on that item
                        output_group = recursively_load_hdf5((key, item))
                        output_group_items.append(output_group)
                # Create output group and set output attributes
                return HDF5Group(
                    name=name,
                    items=output_group_items,
                    attributes={key: val for key, val in group.attrs.items()},
                )
            output_group = recursively_load_hdf5(parent_name_and_group)

        return output_group
