from pathlib import Path
from typing import Optional

import numpy as np


class DATToHDF5Converter:
    """Converter for DAT files to HDF5 format."""

    def read_dat_file(self, input_file_path: Path) -> Optional[np.ndarray]:
        """Read data from a .dat file.

        Args:
            input_file_path (Path): Path to the .dat file to read.

        Returns:
            Optional[np.ndarray]: Array containing the data from the .dat file, or None if file cannot be read.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a .dat file.
        """
        if not input_file_path.exists():
            raise FileNotFoundError(f"The file at {input_file_path} does not exist.")

        if input_file_path.suffix.lower() != ".dat":
            raise ValueError(f"Expected a .dat file, but got {input_file_path.suffix}")

        try:
            # Read the binary data from the .dat file
            with open(input_file_path, "rb") as file:
                data = np.fromfile(file, dtype=np.float32)  # Assuming float32 data type
            return data
        except Exception as e:
            print(f"Error reading .dat file: {str(e)}")
            return None
