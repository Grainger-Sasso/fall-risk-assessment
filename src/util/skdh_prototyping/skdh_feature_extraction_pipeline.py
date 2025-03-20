import datetime
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skdh
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


class SKDHFeatureExtractionPipeline:
    def __init__(self):
        self.file_builder = IMUDataFileBuilder()
        self.model_builder = IMUDataBuilder()
        self.file_writer = HDF5FileWriter()
        self.file_reader = HDF5FileReader()

    def extract_metrics(self, input_file_path: Path, output_file_path: Path) -> Path:
        h5_file: HDF5Group = self.file_reader.read(input_file_path)
        imu_data: IMUData = self.model_builder.build(h5_file)
        pipeline: skdh.Pipeline = self._build_pipeline(output_file_path)
        sensor_data: SensorData = imu_data.data[0].data[0]
        time = sensor_data.time
        # Format is expected to be np array of shape (N, 3) where 3 is three axes: X, Y, Z

        accel = np.column_stack(
            (
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.X).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Y).data,
                sensor_data.get_data_by_sensor_axis(SensorCoordinateSystem.Z).data,
            )
        )
        res = pipeline.run(time=time, accel=accel, height=1.52)

    def _build_pipeline(self, output_file_path: Path) -> skdh.Pipeline:
        """Extracts features

        Args:
            data (_type_): _description_
        """
        """
        pipeline = skdh.Pipeline()
        pipeline.add(skdh.io.ReadCsv(...))  
        pipeline.add(skdh.preprocessing.GetDayWindowIndices(...))  

        # Critical addition for older adult data:
        pipeline.add(skdh.preprocessing.DetectWear(  
            method="van_hees_2013",  # Gold standard for free-living studies  
            min_off_time=90  # 90-min threshold for non-wear (optimized for elders)  
        ))  

        pipeline.add(skdh.preprocessing.CalibrateAccelerometer(...))  
        pipeline.add(skdh.context.PredictGaitLumbarLgbm(...))  
        pipeline.add(skdh.gait.GaitLumbar(...))  
        """
        pipeline = skdh.Pipeline()
        pipeline.add(skdh.preprocessing.GetDayWindowIndices(bases=[0], periods=[24]))
        pipeline.add(skdh.preprocessing.CalibrateAccelerometer())
        pipeline.add(skdh.context.PredictGaitLumbarLgbm())
        pipeline.add(
            skdh.gait.GaitLumbar(),  # default parameters
            save_file=str(
                output_file_path
            ),  # automatically save gait results to a file
            # this will use the CSV file name as the start of the output file name
        )
        return pipeline


def main():
    test_file_path = Path(
        "/Users/graingersasso/Desktop/fafra_testing/test_data/fafra_data/dummy_test_data/test_cs_013_imu_data.h5"
    )
    output_path = Path(
        "/Users/graingersasso/Desktop/fafra_testing/test_data/fafra_data/test_features/test_gait_results.csv"
    )
    extractor = SKDHFeatureExtractionPipeline()
    extractor.extract_metrics(test_file_path, output_path)


if __name__ == "__main__":
    main()
