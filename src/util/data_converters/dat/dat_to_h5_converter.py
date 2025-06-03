import time
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb

from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields


class DATToHDF5Converter:

    SAMPLING_RATE = 100.0
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

    def read_dat_record_wfdb(self, path: Path):
        """Reads and parses data records for LTMM dataset

        Args:
            path (Path): Path the folder containing the LTMM data records + the record name
                e.g. path/to/files/CO001

        Returns:
            _type_: Dictionary of 3D accelerometer data with time axis
        """
        signals, fields = wfdb.rdsamp(path)
        units = fields["units"]  # [g, g, g, deg/s, deg/s, deg/s]
        field_names = fields[
            "sig_name"
        ]  # ['v-acceleration', 'ml-acceleration', 'ap-acceleration', 'yaw-velocity', 'pitch-velocity', 'roll-velocity']
        data = np.array(signals)
        data = np.float16(data)
        v_acc_data = np.array(data.T[0])
        ml_acc_data = np.array(data.T[1])
        ap_acc_data = np.array(data.T[2])
        data = {
            AnatomicalCoordinateSystem.VERTICAL: v_acc_data,
            AnatomicalCoordinateSystem.MEDIOLATERAL: ml_acc_data,
            AnatomicalCoordinateSystem.ANTEROPOSTERIOR: ap_acc_data,
        }

        cur_time = time.time()
        time_axis = np.linspace(
            cur_time,
            (len(v_acc_data) / int(DATToHDF5Converter.SAMPLING_RATE)) + cur_time,
            len(v_acc_data),
        )
        data[IMUDataFields.TIME] = time_axis
        return data


def main():
    path = Path(
        "/Users/graingersasso/Desktop/fafra_data/raw_data/ltmm/long-term-movement-monitoring-database-1.0.0/CO001"
    )
    converter = DATToHDF5Converter()
    data = converter.read_dat_record_wfdb(path)
    print("f")


if __name__ == "__main__":
    main()
