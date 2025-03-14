import datetime
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygt3x.reader import FileReader


class GT3XToH5Converter:
    """Converter for GT3X accelerometer files to HDF5 format"""

    def read_gt3x(self, file_path: Path) -> Tuple[pd.DataFrame, Dict]:
        """Read GT3X file and extract accelerometer data and metadata

        Args:
            file_path (Path): Path to GT3X file

        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame containing accelerometer data and
                dictionary containing metadata

        Raises:
            ValueError: If file does not exist or is not a GT3X file
            zipfile.BadZipFile: If file is corrupted
        """
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        if file_path.suffix.lower() != ".gt3x":
            raise ValueError(f"File is not a GT3X file: {file_path}")

        try:
            with FileReader(str(file_path)) as gt3x_file:
                accelerometer_data = gt3x_file.acceleration
                downsampled_data = self._downsample_data(accelerometer_data, 10)
                self.plot_triaxial_data_with_idle_highlight(
                    downsampled_data[:, 0],
                    downsampled_data[:, 1],
                    downsampled_data[:, 2],
                    downsampled_data[:, 3],
                    downsampled_data[:, 4],
                )
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

        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(f"File is corrupted: {file_path}")

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


def main():
    twenty_min_recording_path = Path(
        "/Users/graingersasso/Desktop/upstate_data/test_sample/FaFRA_CS_013/MOS2E20210841 (2024-12-13).gt3x"
    )
    multidaty_recording_path = Path(
        "/Users/graingersasso/Desktop/upstate_data/test_sample/FaFRA_CS_013/Activity Data/MOS2E20210841 (2024-12-19).gt3x"
    )
    gt3x_converter = GT3XToH5Converter()
    data, metadata = gt3x_converter.read_gt3x(multidaty_recording_path)


if __name__ == "__main__":
    main()
