import datetime
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skdh
from pygt3x.reader import FileReader
from skdh.gait.gait_metrics import gait_metrics

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
from src.util.skdh_prototyping.gait_feature_keys import GaitFeatureKeys
from src.util.skdh_prototyping.output_gait_feature_keys import OutputGaitFeatureKeys


class SKDHFeatureExtractionPipeline:
    def __init__(self):
        self.file_builder = IMUDataFileBuilder()
        self.model_builder = IMUDataBuilder()
        self.file_writer = HDF5FileWriter()
        self.file_reader = HDF5FileReader()
        self.gait_res_key = "GaitLumbar"
        self.event_gait_metrics: List[GaitFeatureKeys] = [
            GaitFeatureKeys.STRIDE_TIME,
            GaitFeatureKeys.STRIDE_TIME_ASYMMETRY,
            GaitFeatureKeys.STANCE_TIME,
            GaitFeatureKeys.STANCE_TIME_ASYMMETRY,
            GaitFeatureKeys.SWING_TIME,
            GaitFeatureKeys.SWING_TIME_ASYMMETRY,
            GaitFeatureKeys.STEP_TIME,
            GaitFeatureKeys.STEP_TIME_ASYMMETRY,
            GaitFeatureKeys.INITIAL_DOUBLE_SUPPORT,
            GaitFeatureKeys.INITIAL_DOUBLE_SUPPORT_ASYMMETRY,
            GaitFeatureKeys.TERMINAL_DOUBLE_SUPPORT,
            GaitFeatureKeys.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY,
            GaitFeatureKeys.DOUBLE_SUPPORT,
            GaitFeatureKeys.DOUBLE_SUPPORT_ASYMMETRY,
            GaitFeatureKeys.SINGLE_SUPPORT,
            GaitFeatureKeys.SINGLE_SUPPORT_ASYMMETRY,
            GaitFeatureKeys.M2_DELTA_H,
            GaitFeatureKeys.M2_DELTA_H_PRIME,
            GaitFeatureKeys.STEP_LENGTH,
            GaitFeatureKeys.STEP_LENGTH_ASYMMETRY,
            GaitFeatureKeys.STRIDE_LENGTH,
            GaitFeatureKeys.STRIDE_LENGTH_ASYMMETRY,
            GaitFeatureKeys.GAIT_SPEED,
            GaitFeatureKeys.GAIT_SPEED_ASYMMETRY,
            GaitFeatureKeys.CADENCE,
            GaitFeatureKeys.M1_DELTA_H,
            GaitFeatureKeys.STEP_LENGTH_M1,
            GaitFeatureKeys.STEP_LENGTH_M1_ASYMMETRY,
            GaitFeatureKeys.STRIDE_LENGTH_M1,
            GaitFeatureKeys.STRIDE_LENGTH_M1_ASYMMETRY,
            GaitFeatureKeys.GAIT_SPEED_M1,
            GaitFeatureKeys.GAIT_SPEED_M1_ASYMMETRY,
            GaitFeatureKeys.INTRA_STEP_COVARIANCE_V,
            GaitFeatureKeys.INTRA_STRIDE_COVARIANCE_V,
            GaitFeatureKeys.HARMONIC_RATIO_V,
            GaitFeatureKeys.STRIDE_SPARC,
        ]
        self.bout_gait_metrics: List[GaitFeatureKeys] = [
            GaitFeatureKeys.BOUT_DURATION,
            GaitFeatureKeys.BOUT_STEPS,
            GaitFeatureKeys.GAIT_CYCLES,
            GaitFeatureKeys.DEBUG_MEAN_STEP_FREQ,
            GaitFeatureKeys.BOUT_PHASE_COORDINATION_INDEX,
            GaitFeatureKeys.BOUT_GAIT_SYMMETRY_INDEX,
            GaitFeatureKeys.BOUT_STEP_REGULARITY_V,
            GaitFeatureKeys.BOUT_STRIDE_REGULARITY_V,
            GaitFeatureKeys.BOUT_AUTOCOVARIANCE_SYMMETRY_V,
            GaitFeatureKeys.BOUT_REGULARITY_INDEX_V,
        ]

    def extract_metrics(
        self, input_file_path: Path, output_file_path: Optional[Path] = None
    ) -> Path:
        h5_file: HDF5Group = self.file_reader.read(input_file_path)
        imu_data: IMUData = self.model_builder.build(h5_file)
        del h5_file
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
        gait_res: Dict = res[self.gait_res_key]
        self.aggregate_multi_day_metrics(gait_res)

    def aggregate_multi_day_metrics(
        self, gait_res: Dict
    ) -> List[Dict[GaitFeatureKeys, float]]:
        # Init results dictionary (dayN - boutN)
        multi_day_metrics: List[Dict[GaitFeatureKeys, float]] = []
        # Initialize pointers for to traverse days and bouts
        day_start_ix = 0
        day_n = 1
        # Reference bout numbers and days from resutls
        day_n_list = gait_res[GaitFeatureKeys.DAY_N.value]
        # Traverse days
        while day_start_ix < len(day_n_list):
            day_end_ix = day_start_ix
            # Traverse days to find end day index
            while day_n_list[day_end_ix] == day_n and day_end_ix < len(day_n_list):
                day_end_ix += 1
            # Aggregate bouts, add to result
            multi_day_metrics.extend(
                self.aggregate_single_day_metrics(gait_res, day_start_ix, day_end_ix)
            )
            # Increment day_n and day_start_ix
            day_n += 1
            day_start_ix = day_end_ix
        # Return results
        return multi_day_metrics

    def aggregate_single_day_metrics(
        self, gait_res, day_start_ix: int, day_end_ix: int
    ) -> List[Dict[GaitFeatureKeys, float]]:
        single_day_metrics = []
        bout_n_list = gait_res[GaitFeatureKeys.BOUT_N.value]
        bout_start_ix = day_start_ix
        bout_n = 1
        while bout_start_ix < len(bout_n_list):
            bout_metrics = {}
            bout_end_ix = bout_start_ix
            while bout_n_list[bout_end_ix] == bout_n and bout_end_ix < len(bout_n_list):
                bout_end_ix += 1
            for event_metric in self.event_gait_metrics:
                bout_metrics[event_metric] = np.mean(
                    gait_res[event_metric.value][bout_start_ix:bout_end_ix]
                )
            for bout_metric in self.bout_gait_metrics:
                bout_metrics[bout_metric] = gait_res[bout_metric.value][bout_start_ix]
            single_day_metrics.append(bout_metrics)
            bout_n += 1
            bout_start_ix = bout_end_ix
        return single_day_metrics

    def aggregate_bout_level_metrics(self):
        # Define the specific metrics in the gait results which are to be aggregated
        # Define the descriptive statistics to be used in aggregating metrics
        # For each one of those gait metrics and descriptive stats, generate aggregate data objects
        pass

    def export_aggregate_metrics(self):
        pass

    def _build_pipeline(self, output_file_path: Optional[Path]) -> skdh.Pipeline:
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
        if output_file_path:
            pipeline.add(
                skdh.gait.GaitLumbar(),  # default parameters
                save_file=str(
                    output_file_path
                ),  # automatically save gait results to a file
                # this will use the CSV file name as the start of the output file name
            )
        else:
            pipeline.add(skdh.gait.GaitLumbar())
        return pipeline


def main():
    test_file_path = Path(
        "/Users/graingersasso/Desktop/fafra_testing/test_data/fafra_data/dummy_test_data/test_cs_013_imu_data.h5"
    )
    output_path = Path(
        "/Users/graingersasso/Desktop/fafra_testing/test_data/fafra_data/test_features/test_gait_results.csv"
    )
    extractor = SKDHFeatureExtractionPipeline()
    # extractor.extract_metrics(test_file_path, output_path)
    extractor.extract_metrics(test_file_path)


if __name__ == "__main__":
    main()
