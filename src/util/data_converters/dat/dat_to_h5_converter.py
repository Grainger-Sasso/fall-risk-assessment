import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import wfdb

from src.data_io.builders.file_builders.data.imu.imu_data_file_builder import (
    IMUDataFileBuilder,
)
from src.data_io.builders.model_builders.data.imu.imu_data_builder import IMUDataBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataFileNames,
)
from src.data_io.model_fields.data.imu.imu_data_fields import IMUDataFields
from src.data_io.model_fields.data.user.clinical_demographic_data_fields import (
    ClinicalDemographicDataFields,
)
from src.data_io.model_fields.data.user.user_data_fields import UserDataFields
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.data_io.read_write.readers.hdf5.hdf5_file_reader import HDF5FileReader
from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter
from src.data_io.read_write.writers.json.json_dict_file_writer import JSONDictFileWriter
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
    FallerStatus,
    Sex,
)
from src.data_model.data.user.user_data import UserData
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.identifier import IdentifierGenerator
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)
from src.identifiers.user.clinical_identifier import ClinicalIdentifier
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
        self.imu_id_gen = IdentifierGenerator("imu", IMUDataIdentifier)
        self.file_builder = IMUDataFileBuilder()
        self.model_builder = IMUDataBuilder()
        self.file_writer = HDF5FileWriter()
        self.file_reader = HDF5FileReader()
        self.csv_file_reader = CSVFileReader()
        self.json_dict_writer = JSONDictFileWriter()
        self.imu_data_exporter = IMUDataFileExporter()
        self.p_ids = [
            "CO001",
            "CO002",
            "CO003",
            "CO004",
            "CO005",
            "CO006",
            "CO007",
            "CO008",
            "CO009",
            "CO010",
            "CO011",
            "CO012",
            "CO013",
            "CO014",
            "CO015",
            "CO016",
            "CO017",
            "CO018",
            "CO019",
            "CO020",
            "CO021",
            "CO022",
            "CO023",
            "CO024",
            "CO025",
            "CO027",
            "CO028",
            "CO029",
            "CO030",
            "CO031",
            "CO032",
            "CO035",
            "CO036",
            "CO037",
            "CO038",
            "CO039",
            "CO040",
            "CO041",
            "CO042",
            "CO044",
            "FL001",
            "FL004",
            "FL005",
            "FL006",
            "FL007",
            "FL008",
            "FL009",
            "FL010",
            "FL011",
            "FL014",
            "FL015",
            "FL016",
            "FL018",
            "FL019",
            "FL020",
            "FL021",
            "FL022",
            "FL023",
            "FL024",
            "FL025",
            "FL026",
            "FL027",
            "FL028",
            "FL029",
            "FL030",
            "FL031",
            "FL032",
            "FL033",
            "FL034",
            "FL035",
            "FL036",
        ]
        self.lab_walks_pids = [
            "CO001",
            "CO002",
            "CO003",
            "CO004",
            "CO005",
            "CO006",
            "CO007",
            "CO008",
            "CO009",
            "CO010",
            "CO011",
            "CO013",
            "CO014",
            "CO015",
            "CO016",
            "CO017",
            "CO018",
            "CO019",
            "CO020",
            "CO021",
            "CO022",
            "CO023",
            "CO024",
            "CO025",
            "CO026",
            "CO027",
            "CO028",
            "CO029",
            "CO030",
            "CO031",
            "CO032",
            "CO033",
            "CO034",
            "CO035",
            "CO036",
            "CO040",
            "CO041",
            "CO042",
            "FL001",
            "FL003",
            "FL004",
            "FL005",
            "FL006",
            "FL007",
            "FL008",
            "FL009",
            "FL010",
            "FL011",
            "FL013",
            "FL015",
            "FL016",
            "FL017",
            "FL018",
            "FL019",
            "FL020",
            "FL021",
            "FL022",
            "FL023",
            "FL024",
            "FL025",
            "FL026",
            "FL027",
            "FL028",
            "FL030",
            "FL031",
            "FL032",
            "FL033",
            "FL034",
            "FL035",
            "FL036",
            "FL037",
            "FL038",
            "FL039",
        ]

    ### Demo data conversion
    def read_xlsx_to_dict(self, file_path: Path) -> Dict[str, List[Any]]:
        """Read an XLSX file into a dictionary mapping column headers to arrays of values.

        Args:
            file_path (Path): Path to the XLSX file

        Returns:
            Dict[str, List[Any]]: Dictionary where keys are column headers and values are lists of data

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file is not an XLSX file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"The file at {file_path} does not exist.")

        if file_path.suffix.lower() != ".xlsx":
            raise ValueError(f"Expected an XLSX file, but got {file_path.suffix}")

        try:
            # Read the Excel file
            df = pd.read_excel(file_path)

            # Convert DataFrame to dictionary of lists
            data_dict = {column: df[column].tolist() for column in df.columns}

            return data_dict
        except Exception as e:
            print(f"Error reading XLSX file: {str(e)}")
            return {}

    def build_user_data(self, demo_data_path: Path, output_path: Path):
        demo_data = self.read_xlsx_to_dict(demo_data_path)

        # For each entry in the data, create
        ix = 0
        num_entries = len(demo_data["Participant ID"])
        while ix < num_entries:
            p_id = demo_data["Participant ID"][ix].replace("-", "")
            age = demo_data["Age"][ix]
            sex = demo_data["Sex (Male-0; Female-1)"][ix]
            faller = demo_data["Faller Status"][ix]
            if sex == 1:
                sex = Sex.FEMALE
            else:
                sex = Sex.MALE
            if faller:
                faller = FallerStatus.FALLER
            else:
                faller = FallerStatus.NON_FALLER
            # Build user json file
            user_data_json_dict = JSONDictFile(
                {UserDataFields.USER_DATA_IDENTIFIER.value: p_id}
            )
            # Build clin demo data json file
            clin_data_json_dict = JSONDictFile(
                {
                    ClinicalDemographicDataFields.NAME.value: "",
                    ClinicalDemographicDataFields.AGE.value: age,
                    ClinicalDemographicDataFields.SEX.value: sex.value,
                    ClinicalDemographicDataFields.WEIGHT.value: 71.98,
                    ClinicalDemographicDataFields.HEIGHT.value: 1.62,
                    ClinicalDemographicDataFields.IDENTIFIER.value: p_id,
                    ClinicalDemographicDataFields.FALLER_STATUS.value: faller.value,
                }
            )
            # Gen directory for user data output
            output_subdir_path = os.path.join(output_path, "user_" + p_id)
            #
            os.makedirs(output_subdir_path, exist_ok=True)
            user_data_path = os.path.join(
                output_subdir_path, UserDataFileNames.USER_DATA.value + ".json"
            )
            clin_data_path = os.path.join(
                output_subdir_path,
                UserDataFileNames.CLININCAL_DEMOGRAPHIC_DATA.value + ".json",
            )
            s1, e1 = self.json_dict_writer.write(user_data_path, user_data_json_dict)
            s2, e2 = self.json_dict_writer.write(clin_data_path, clin_data_json_dict)
            print(s1, e1)
            ix += 1

    def build_user_data_object(
        male_status_1: bool, faller: int, age: float, p_id: str, sex: float
    ):
        if male_status_1:
            if sex == 1:
                sex = Sex.MALE
            else:
                sex = Sex.FEMALE
        else:
            if sex == 1:
                sex = Sex.FEMALE
            else:
                sex = Sex.MALE
        if faller:
            faller = FallerStatus.FALLER
        else:
            faller = FallerStatus.NON_FALLER
        clin_id = ClinicalIdentifier(p_id)
        clin_demo_data = ClinicalDemographicData(
            "", age, sex, 0.0, 0.0, clin_id, faller
        )
        user_id = UserIdentifier(p_id)
        return UserData(user_id, clin_demo_data)

    ### IMU Data conversion
    def convert_all_dat_to_h5(
        self, dat_dir_path: Path, output_path: Path, lab_walks: bool = False
    ):
        p_id_to_imu_id_map = {}
        imu_id_to_path_map = {}
        p_ids = self.lab_walks_pids if lab_walks else self.p_ids
        for p_id in p_ids:
            # NOTE THE FILE PATH NEEDS TO BE PROVIDED AS DIRECTORY + P_ID (lib assumes .dat and .hea file format)
            suffix = p_id if not lab_walks else p_id.lower() + '_base'
            file_path = Path(os.path.join(dat_dir_path, suffix))
            data = self.read_dat_record_wfdb(file_path)
            print(f"Read in file: {p_id}")
            imu_data: IMUData = self.convert_to_imu_data(data, p_id)
            p_id_to_imu_id_map[p_id] = imu_data.metadata.imu_data_identifier.value
            print(f"Converted file: {p_id}")
            exported_path = self.imu_data_exporter.export_data(output_path, imu_data)
            imu_id_to_path_map[imu_data.metadata.imu_data_identifier.value] = str(exported_path)
            print(f"Exported file: {p_id}")
        return p_id_to_imu_id_map, imu_id_to_path_map

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

    def test_read_converted_file(self, path: Path):
        h5_file: HDF5Group = self.file_reader.read(path)
        return self.model_builder.build(h5_file)

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
        idle_mask = np.array(
            [0 for _ in range(len(data[AnatomicalCoordinateSystem.ANTEROPOSTERIOR]))]
        )
        return SensorData(
            data=[x_axis, y_axis, z_axis],
            time=time,
            idle_mask=idle_mask,
            metadata=sensor_metadata,
        )


def main():
    # ### Converts single LTMM data file to H5 file and writes to specified path
    # path = Path(
    #     "/Users/graingersasso/Desktop/fafra_data/raw_data/ltmm/long-term-movement-monitoring-database-1.0.0/CO001"
    # )
    # output_path = Path(
    #     "/Users/graingersasso/Desktop/fafra_data/raw_data/ltmm_h5/test.h5"
    # )
    # converter = DATToHDF5Converter()
    # data = converter.read_dat_record_wfdb(path)
    # imu_data = converter.convert_to_imu_data(data, "dummy_user_id")
    # converter.export_imu_data_to_h5(imu_data, output_path)
    # converted_imu_data = converter.test_read_converted_file(output_path)

    # ### Converts CSV file with Demo data into JSON files in subdirs by ID
    # demo_data_path = Path(
    #     "/Users/graingersasso/Desktop/fafra_data/raw_data/ltmm/long-term-movement-monitoring-database-1.0.0/demo_data_essential.xlsx"
    # )
    # user_output_path = Path(
    #     "/Users/graingersasso/Desktop/fafra_data/converted_data/ltmm/user_data"
    # )
    # converter = DATToHDF5Converter()
    # demo_data = converter.build_user_data(demo_data_path, user_output_path)

    ### Converts IMU to h5
    converter = DATToHDF5Converter()
    imu_data_dir = Path(
        "/Users/graingersasso/Desktop/fafra/fafra_data/raw_data/ltmm/LabWalks"
    )
    p_id_to_imu_id_map, imu_id_to_path_map = converter.convert_all_dat_to_h5(
        imu_data_dir,
        Path("/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_lab_walks_2026_05_09"),
        True
    )
    print(p_id_to_imu_id_map)
    print(imu_id_to_path_map)
    print("f")


if __name__ == "__main__":
    main()
