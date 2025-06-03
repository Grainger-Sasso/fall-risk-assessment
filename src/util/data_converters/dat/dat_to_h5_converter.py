import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import wfdb

from src.data_io.builders.file_builders.data.imu.imu_data_file_builder import (
    IMUDataFileBuilder,
)
from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
from src.id_generator.imu.imu_data_identifier_generator import (
    IMUDataIdentifierGenerator,
)
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


class DATToHDF5Converter:

    SAMPLING_RATE = 100.0
    """Converter for DAT files to HDF5 format."""

    def __init__(self):
        self.imu_id_gen = IMUDataIdentifierGenerator()
        self.file_builder = IMUDataFileBuilder()
        self.model_builder = IMUDataBuilder()
        self.file_writer = HDF5FileWriter()
        self.file_reader = HDF5FileReader()

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

    def read_dat_record_wfdb(self, path: Path) -> Dict[Any, Any]:
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

    def convert_to_imu_data(self, data, user_id: str) -> IMUData:
        epoch_imu_data_list: List[EpochIMUData] = self._build_epoch_data(data)
        imu_data_id: IMUDataIdentifier = self.imu_id_gen.generate_identifier()
        user_data_id: UserIdentifier = UserIdentifier(user_id)
        inst_id: InstrumentIdentifier = InstrumentIdentifier("placeholder", "001")
        metadata: IMUMetadata = IMUMetadata(
            imu_data_identifier=imu_data_id,
            user_identifier=user_data_id,
            instrument_identifier=inst_id,
        )
        start_time = data[IMUDataFields.TIME][0]
        end_time = data[IMUDataFields.TIME][-1]
        return IMUData(
            data=epoch_imu_data_list,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
        )

    def export_imu_data_to_h5(self, data: IMUData, output_path: Path):
        group: HDF5Group = self.file_builder.build(data)
        self.file_writer.write(output_path, group)
        return

    def _build_epoch_data(self, data) -> List[EpochIMUData]:
        sensor_data = self._build_sensor_data(data)
        start_time = data[IMUDataFields.TIME][0]
        end_time = data[IMUDataFields.TIME][-1]
        return [
            EpochIMUData(
                data=[sensor_data], epoch_start_time=start_time, epoch_end_time=end_time
            )
        ]

    def _build_sensor_data(self, data) -> List[SensorData]:
        x_axis = UniaxialSensorData(
            anatomical_axis=AnatomicalAxis(AnatomicalCoordinateSystem.VERTICAL),
            sensor_axis=SensorAxis(SensorCoordinateSystem.X),
            data=data[AnatomicalCoordinateSystem.VERTICAL],
        )
        y_axis = UniaxialSensorData(
            anatomical_axis=AnatomicalAxis(AnatomicalCoordinateSystem.MEDIOLATERAL),
            sensor_axis=SensorAxis(SensorCoordinateSystem.Y),
            data=data[AnatomicalCoordinateSystem.MEDIOLATERAL],
        )
        z_axis = UniaxialSensorData(
            anatomical_axis=AnatomicalAxis(AnatomicalCoordinateSystem.ANTEROPOSTERIOR),
            sensor_axis=SensorAxis(SensorCoordinateSystem.Z),
            data=data[AnatomicalCoordinateSystem.ANTEROPOSTERIOR],
        )
        sensor_metadata = SensorMetadata(
            sensor_type=SensorType.ACCELEROMETER,
            sampling_rate=DATToHDF5Converter.SAMPLING_RATE,
            unit="g",
        )
        time = data[IMUDataFields.TIME]
        idle_mask = np.array([0 for _ in range(len(data[AnatomicalCoordinateSystem.ANTEROPOSTERIOR]))])
        return SensorData(
            data=[x_axis, y_axis, z_axis],
            time=time,
            idle_mask=idle_mask,
            metadata=sensor_metadata,
        )

    def test_read_converted_file(self, path: Path):
        h5_file: HDF5Group = self.file_reader.read(path)
        return self.model_builder.build(h5_file)


def main():
    path = Path(
        "/Users/graingersasso/Desktop/fafra_data/raw_data/ltmm/long-term-movement-monitoring-database-1.0.0/CO001"
    )
    output_path = Path(
        "/Users/graingersasso/Desktop/fafra_data/raw_data/ltmm_h5/test.h5"
    )
    converter = DATToHDF5Converter()
    data = converter.read_dat_record_wfdb(path)
    imu_data = converter.convert_to_imu_data(data, "dummy_user_id")
    converter.export_imu_data_to_h5(imu_data, output_path)
    converted_imu_data = converter.test_read_converted_file(output_path)


if __name__ == "__main__":
    main()
