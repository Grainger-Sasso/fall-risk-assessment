import datetime
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygt3x.reader import FileReader

from src.data_io.builders.file_builders.data.imu.imu_data_file_builder import (
    IMUDataFileBuilder,
)
from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)


class GT3XToH5Converter:
    """Converter for GT3X accelerometer files to HDF5 format"""

    def __init__(self):
        self.file_builder = IMUDataFileBuilder()
        self.model_builder = IMUDataBuilder()
        self.file_writer = HDF5FileWriter()
        self.file_reader = HDF5FileReader()
        # Scale factor to be applied to accelerometer data to convert to units of g
        self.scale_factor = 256.0
        self.sampling_rate = 100.0
        self.user_id = "dummy_user_id"

    def convert_gt3x_to_h5(
        self,
        input_file_path: Path,
        output_file_path: Path,
        scale_factor: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Convert GT3X file to H5 format with optional parameters

        Args:
            input_file_path (Path): Path to GT3X file
            output_file_path (Path): Path to output H5 file
            scale_factor (Optional[float], optional): Custom scale factor. Defaults to None.
            sampling_rate (Optional[float], optional): Custom sampling rate. Defaults to None.
            user_id (Optional[str], optional): Custom user identifier. Defaults to None.

        Raises:
            ValueError: If file does not exist or is not a GT3X file
            zipfile.BadZipFile: If file is corrupted
        """
        if not input_file_path.exists():
            raise ValueError(f"File does not exist: {input_file_path}")

        if input_file_path.suffix.lower() != ".gt3x":
            raise ValueError(f"File is not a GT3X file: {input_file_path}")

        try:
            # Use provided values or fall back to defaults
            self.scale_factor = scale_factor or 256.0
            self.sampling_rate = sampling_rate or 100.0
            self.user_id = user_id or "dummy_user_id"

            with FileReader(str(input_file_path)) as gt3x_file:
                print("########## GT3X #############")
                print(gt3x_file.acceleration[0, 0])
                print(gt3x_file.acceleration[-1, 0])
                print(len(gt3x_file.acceleration[:, 1]))
                print(gt3x_file.acceleration[:10, 1])
                acc_data = gt3x_file.to_pandas(calibrate=False)

                h5_file: HDF5Group = self.build_h5_file(gt3x_file)
                self.write_h5_file(h5_file, output_file_path)

                # downsampled_data = self._downsample_data(accelerometer_data, 10)
                # self.plot_triaxial_data_with_idle_highlight(
                #     downsampled_data[:, 0],
                #     downsampled_data[:, 1],
                #     downsampled_data[:, 2],
                #     downsampled_data[:, 3],
                #     downsampled_data[:, 4],
                # )

        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(f"File is corrupted: {input_file_path}")

    def build_h5_file(self, gt3x_file: FileReader) -> HDF5Group:
        if gt3x_file.info.acceleration_scale != self.scale_factor:
            raise ValueError(
                "Scale factor in metadata doesn't match expected scale factor"
            )
        sensor_data: SensorData = self._build_sensor_data(gt3x_file)
        start_time = sensor_data.time[0]
        end_time = sensor_data.time[-1]
        epoch_imu_data = EpochIMUData(
            data=[sensor_data], epoch_start_time=start_time, epoch_end_time=end_time
        )
        # Build metadata
        info = gt3x_file.info
        sensor_name = "actigraph_" + info.device_type
        sensor_serial_number = info.serial_number
        imu_metadata = IMUMetadata(
            imu_data_identifier=IMUDataIdentifier("dummy_imu_id"),
            user_identifier=UserIdentifier(self.user_id),
            instrument_identifier=InstrumentIdentifier(
                sensor_name, sensor_serial_number
            ),
        )
        imu_data = IMUData(
            data=[epoch_imu_data],
            metadata=imu_metadata,
            start_time=start_time,
            end_time=end_time,
        )
        return self.file_builder.build(imu_data)

    def _build_sensor_data(self, gt3x_file: FileReader) -> SensorData:
        # Get time and idle mask from data
        time_data: np.ndarray = gt3x_file.acceleration[:, 0]
        idle_data: np.ndarray = gt3x_file.acceleration[:, 4]
        # Get triaxial accelerometer data and convert to Uniaxial data
        # MD
        x_data: np.ndarray = gt3x_file.acceleration[:, 1] / self.scale_factor
        # Vertical
        y_data: np.ndarray = gt3x_file.acceleration[:, 2] / self.scale_factor
        # AP
        z_data: np.ndarray = gt3x_file.acceleration[:, 3] / self.scale_factor
        x_uniaxial = UniaxialSensorData(
            anatomical_axis=AnatomicalAxis(AnatomicalCoordinateSystem.MEDIOLATERAL),
            sensor_axis=SensorAxis(SensorCoordinateSystem.X),
            data=x_data,
        )
        y_uniaxial = UniaxialSensorData(
            anatomical_axis=AnatomicalAxis(AnatomicalCoordinateSystem.VERTICAL),
            sensor_axis=SensorAxis(SensorCoordinateSystem.Y),
            data=y_data,
        )
        z_uniaxial = UniaxialSensorData(
            anatomical_axis=AnatomicalAxis(AnatomicalCoordinateSystem.ANTEROPOSTERIOR),
            sensor_axis=SensorAxis(SensorCoordinateSystem.Z),
            data=z_data,
        )

        sensor_metadata = SensorMetadata(
            sensor_type=SensorType.ACCELEROMETER,
            sampling_rate=self.sampling_rate,
            unit="g",
        )
        return SensorData(
            data=[x_uniaxial, y_uniaxial, z_uniaxial],
            time=time_data,
            idle_mask=idle_data,
            metadata=sensor_metadata,
        )

    def write_h5_file(self, file: HDF5Group, output_file_path):
        self.file_writer.write(output_file_path, file)

    def test_read_converted_file(self, path: Path):
        h5_file: HDF5Group = self.file_reader.read(path)
        imu_data: IMUData = self.model_builder.build(h5_file)
        print("########## h5 #############")
        print(imu_data.data[0].epoch_start_time)
        print(imu_data.data[0].epoch_end_time)
        print(
            len(
                imu_data.data[0]
                .data[0]
                .get_data_by_sensor_axis(SensorCoordinateSystem.X)
                .data
            )
        )
        print(
            imu_data.data[0]
            .data[0]
            .get_data_by_sensor_axis(SensorCoordinateSystem.X)
            .data[:10]
        )
        pass

    def plot_triaxial_data(self, time, x_data, y_data, z_data):
        plt.figure(figsize=(10, 6))
        plt.plot(time, x_data, label="X-axis")
        plt.plot(time, y_data, label="Y-axis")
        plt.plot(time, z_data, label="Z-axis")

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Tri-axial Data")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_triaxial_data_with_idle_highlight(
        self, time, x_data, y_data, z_data, idle
    ):
        """Plot tri-axial data with highlighted sections for idle periods

        Args:
            time: Array of timestamps
            x_data: X-axis acceleration data
            y_data: Y-axis acceleration data
            z_data: Z-axis acceleration data
            idle: Boolean array indicating idle periods
        """
        plt.figure(figsize=(10, 6))

        # Plot the acceleration data
        plt.plot(time, x_data, label="X-axis")
        plt.plot(time, y_data, label="Y-axis")
        plt.plot(time, z_data, label="Z-axis")

        # Find idle regions
        idle_mask = idle != 0
        changes = np.diff(idle_mask.astype(int))
        idle_starts = np.where(changes == 1)[0] + 1
        idle_ends = np.where(changes == -1)[0] + 1

        # Handle case where data starts idle
        if idle_mask[0]:
            idle_starts = np.insert(idle_starts, 0, 0)

        # Handle case where data ends idle
        if idle_mask[-1]:
            idle_ends = np.append(idle_ends, len(time))

        # Add highlighted regions for idle periods
        y_min, y_max = plt.ylim()
        for start, end in zip(idle_starts, idle_ends):
            plt.axvspan(
                time[start],
                time[end],
                color="#ffcccc",  # Light red color
                alpha=0.3,
                label="Idle" if start == idle_starts[0] else "",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Tri-axial Data with Idle Periods")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_triaxial_data_with_gaps_single(
        self,
        time: np.ndarray,
        x_data: np.ndarray,
        y_data: np.ndarray,
        z_data: np.ndarray,
        sleep_axis_data: np.ndarray,
    ):
        """Plot tri-axial data on a single plot with gaps for sleep mode."""
        # Create mask for non-sleep data
        active_mask = sleep_axis_data != 0

        # Split data into segments
        segments = self.split_into_segments(time, active_mask)

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot each segment
        for segment in segments:
            plt.plot(time[segment], x_data[segment], "r-", label="X-axis")
            plt.plot(time[segment], y_data[segment], "g-", label="Y-axis")
            plt.plot(time[segment], z_data[segment], "b-", label="Z-axis")

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Tri-axial Data (Gaps Indicate Sleep Mode)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def split_into_segments(
        self, time: np.ndarray, active_mask: np.ndarray
    ) -> List[np.ndarray]:
        """
        Split data into continuous segments based on active_mask.

        Args:
            time: Time array
            active_mask: Boolean mask where True indicates active data

        Returns:
            List of index arrays for each continuous segment
        """
        # Find where mask changes (edges of segments)
        changes = np.diff(active_mask.astype(int))
        segment_starts = np.where(changes == 1)[0] + 1
        segment_ends = np.where(changes == -1)[0] + 1

        # Handle case where data starts active
        if active_mask[0]:
            segment_starts = np.insert(segment_starts, 0, 0)

        # Handle case where data ends active
        if active_mask[-1]:
            segment_ends = np.append(segment_ends, len(time))

        # Create segments
        segments = []
        for start, end in zip(segment_starts, segment_ends):
            segments.append(np.arange(start, end))

        return segments

    def _parse_metadata(self, info_text: str) -> Dict:
        """Parse metadata from info.txt file

        Args:
            info_text (str): Contents of info.txt file

        Returns:
            Dict: Dictionary containing metadata fields
        """
        metadata = {}
        for line in info_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def _parse_activity_data(self, log_file_path: Path) -> pd.DataFrame:
        """Parse accelerometer data from log.bin file

        Args:
            log_file_path (Path): path to log file

        Returns:
            pd.DataFrame: DataFrame containing accelerometer data with columns:
                - timestamp: Timestamp of measurement
                - x: X-axis acceleration (g)
                - y: Y-axis acceleration (g)
                - z: Z-axis acceleration (g)
        """
        with FileReader(str(log_file_path)) as reader:
            was_idle_sleep_mode_used = reader.idle_sleep_mode_activated
            df = reader.to_pandas()
            return df

        # # Convert binary data to numpy array
        # # Each sample is 6 bytes:
        # # - 2 bytes each for x, y, z acceleration
        # # - Values are signed 16-bit integers in range [-2048, 2047]
        # # - Scale factor is 341.0 counts/g
        # data = np.frombuffer(binary_data, dtype=np.int16)
        # data = data.reshape(-1, 3)

        # # Convert to g units
        # scale_factor = 341.0  # counts per g
        # data = data / scale_factor

        # # Create DataFrame
        # df = pd.DataFrame(data, columns=["x", "y", "z"])

        # # Add timestamp column
        # # Sample rate is typically stored in metadata
        # # For now use default 30 Hz
        # sample_rate = 30.0  # Hz
        # df["timestamp"] = pd.date_range(
        #     start=pd.Timestamp.now(), periods=len(df), freq=f"{1/sample_rate}S"
        # )

        # return df[["timestamp", "x", "y", "z"]]

    def _downsample_data(self, data: np.ndarray, factor: int) -> np.ndarray:
        """Downsample the data by taking means of groups of samples"""
        # Ensure the factor is an integer
        factor = int(factor)

        # Calculate the new length of the downsampled array
        new_length = data.shape[0] // factor

        # Reshape the array to group samples while preserving columns
        reshaped = data[: new_length * factor].reshape(-1, factor, data.shape[1])

        # Calculate the mean of each group
        downsampled = np.mean(reshaped, axis=1)

        return downsampled

    # def convert_using_zipfile(self):
    # Downsample by factor of 10

    # # Split the data into its components
    # time = downsampled_data[:, 0]  # First column is time
    # x_data = downsampled_data[:, 1]  # Second column is x-axis
    # y_data = downsampled_data[:, 2]  # Third column is y-axis
    # z_data = downsampled_data[:, 3]  # Fourth column is z-axis
    # sleep_data = downsampled_data[:, 4]  # Fifth column is sleep mode

    # # Plot the data showing gaps where sensor is in sleep mode
    # self.plot_triaxial_data_with_gaps_single(
    #     time, x_data, y_data, z_data, sleep_data
    # )

    #     print(
    #         datetime.datetime.fromtimestamp(
    #             gt3x_file.acceleration[0][0]
    #         ).strftime("%c")
    #     )
    #     print(
    #         datetime.datetime.fromtimestamp(
    #             gt3x_file.acceleration[-1][0]
    #         ).strftime("%c")
    #     )
    #     print(gt3x_file.acceleration[1][0] - gt3x_file.acceleration[0][0])
    #     print(gt3x_file)

    # with zipfile.ZipFile(file_path, "r") as zip_ref:
    #     # Extract metadata from info.txt
    #     metadata = self._parse_metadata(
    #         zip_ref.read("info.txt").decode("utf-8")
    #     )

    #     # Extract accelerometer data from log.bin
    #     data = self._parse_activity_data(zip_ref.read("log.bin"))

    #     return data, metadata


def main():
    twenty_min_recording_path = Path(
        "/Users/graingersasso/Desktop/upstate_data/test_sample/FaFRA_CS_013/MOS2E20210841 (2024-12-13).gt3x"
    )
    # multidaty_recording_path = Path(
    #     "/Users/graingersasso/Desktop/upstate_data/test_sample/FaFRA_CS_013/Activity Data/MOS2E20210841 (2024-12-19).gt3x"
    # )
    multidaty_recording_path = Path(
        "/Users/graingersasso/Desktop/fafra_testing/test_data/upstate_data/FaFRA_CS_013/Activity Data/MOS2E20210841 (2024-12-19).gt3x"
    )
    output_file_path = Path(
        "/Users/graingersasso/Desktop/fafra_testing/test_data/fafra_data/dummy_test_data/test_cs_013_imu_data.h5"
    )

    gt3x_converter = GT3XToH5Converter()
    # gt3x_converter.convert_gt3x_to_h5(multidaty_recording_path, output_file_path)
    gt3x_converter.test_read_converted_file(output_file_path)


if __name__ == "__main__":
    main()
