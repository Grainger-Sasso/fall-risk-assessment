import h5py
from pathlib import Path

from src.data_io.read_write.readers.file_reader import FileReader
from src.data_io.formats.hdf5.group import Group
from src.data_io.formats.hdf5.dataset import Dataset


class HDF5FileReader(FileReader):
    """
    HDF5 file reader.
    """

    def read(self, path: Path) -> Group:
        """Reads data from an HDF5 file and returns them Group format.

        Args:
            path (Path): The path to the HDF5 file.

        Returns:
            Group: A Group containing datasets, groups, and their attributes.
        """
        if not path.exists():
            raise FileNotFoundError(f"The file at {path} does not exist.")

        if not path.suffix in [".h5", ".hdf5"]:
            raise ValueError(
                f"Expected an HDF5 file (.h5 or .hdf5), but got {path.suffix}."
            )
        output_group = Group()
        with h5py.File(path, "r") as file:

            def recursively_load_hdf5(group: h5py.Group, target: Group) -> None:
                """Recursively reads a Group object to an HDF5 group.

                Args:
                    group (h5py.Group): The Group object to read.
                    target (Group): The target group.
                """
                target.name = group.name
                target.attributes = {key: val for key, val in group.attrs.items()}
                target.items = []
                for key, item in group.items():
                    if isinstance(item, h5py.Dataset):
                        dataset = Dataset(
                            name=key,
                            data=item[()],
                            attributes={key: val for key, val in item.attrs.items()},
                        )
                        target.items.append(dataset)
                    elif isinstance(item, h5py.Group):
                        subgroup = Group()
                        recursively_load_hdf5(subgroup, item)
                        target.items.append(subgroup)

            recursively_load_hdf5(file, output_group)
        return output_group
