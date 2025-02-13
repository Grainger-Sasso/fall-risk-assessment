import h5py
from pathlib import Path
from typing import Any, Tuple

from data_io.formats.hdf5.hdf5_group import HDF5Group
from data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.read_write.writers.file_writer import FileWriter


class JHDF5FileWriter(FileWriter):
    """
    HDF5 file writer.
    """

    def write(self, path: Path, group: HDF5Group, **kwargs) -> Tuple[bool, str]:
        """Writes data and attributes from Group and Dataset objects to an HDF5 file.

        Args:
            path (Path): The path to the HDF5 file.
            group (Group): The root Group object to write.

        Returns:
            Tuple[bool, str]: (success flag, error message)
        """
        if path.suffix not in [".h5", ".hdf5"]:
            return (
                False,
                f"Expected an HDF5 file (.h5 or .hdf5), but got {path.suffix}.",
            )

        def recursively_write_hdf5(target: HDF5Group, h5_group: h5py.Group) -> None:
            """Recursively writes a Group object to an HDF5 group.

            Args:
                target (Group): The Group object to write.
                h5_group (h5py.Group): The target HDF5 group.
            """
            # Write attributes
            for key, val in target.attributes.items():
                h5_group.attrs[key] = val

            # Write items
            for item in target.items:
                if isinstance(item, HDF5Dataset):
                    # Create a dataset and write its data and attributes
                    dataset = h5_group.create_dataset(item.name, data=item.data)
                    for key, val in item.attributes.items():
                        dataset.attrs[key] = val
                elif isinstance(item, HDF5Group):
                    # Create a subgroup and recursively write its contents
                    subgroup = h5_group.create_group(item.name)
                    recursively_write_hdf5(item, subgroup)

        try:
            with h5py.File(path, "w") as file:
                recursively_write_hdf5(group, file)
            return True, ""  # Success
        except Exception as e:
            return False, str(e)  # Failure with error message
