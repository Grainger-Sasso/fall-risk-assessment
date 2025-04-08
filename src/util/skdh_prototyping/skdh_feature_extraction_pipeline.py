import datetime
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
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
from src.util.skdh_prototyping.descriptive_statistic_types import (
    DescriptiveStatisticsTypes,
)
from src.util.skdh_prototyping.gait_feature_types import GaitFeatureTypes



class SKDHFeatureExtractionPipeline:
    def __init__(self):
        self.file_builder = IMUDataFileBuilder()
        self.model_builder = IMUDataBuilder()
        self.file_writer = HDF5FileWriter()
        self.file_reader = HDF5FileReader()
        self.gait_res_key = "GaitLumbar"
        self.event_gait_metrics: List[GaitFeatureTypes] = [
            GaitFeatureTypes.STRIDE_TIME,
            GaitFeatureTypes.STRIDE_TIME_ASYMMETRY,
            GaitFeatureTypes.STANCE_TIME,
            GaitFeatureTypes.STANCE_TIME_ASYMMETRY,
            GaitFeatureTypes.SWING_TIME,
            GaitFeatureTypes.SWING_TIME_ASYMMETRY,
            GaitFeatureTypes.STEP_TIME,
            GaitFeatureTypes.STEP_TIME_ASYMMETRY,
            GaitFeatureTypes.INITIAL_DOUBLE_SUPPORT,
            GaitFeatureTypes.INITIAL_DOUBLE_SUPPORT_ASYMMETRY,
            GaitFeatureTypes.TERMINAL_DOUBLE_SUPPORT,
            GaitFeatureTypes.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY,
            GaitFeatureTypes.DOUBLE_SUPPORT,
            GaitFeatureTypes.DOUBLE_SUPPORT_ASYMMETRY,
            GaitFeatureTypes.SINGLE_SUPPORT,
            GaitFeatureTypes.SINGLE_SUPPORT_ASYMMETRY,
            GaitFeatureTypes.M2_DELTA_H,
            GaitFeatureTypes.M2_DELTA_H_PRIME,
            GaitFeatureTypes.STEP_LENGTH,
            GaitFeatureTypes.STEP_LENGTH_ASYMMETRY,
            GaitFeatureTypes.STRIDE_LENGTH,
            GaitFeatureTypes.STRIDE_LENGTH_ASYMMETRY,
            GaitFeatureTypes.GAIT_SPEED,
            GaitFeatureTypes.GAIT_SPEED_ASYMMETRY,
            GaitFeatureTypes.CADENCE,
            GaitFeatureTypes.M1_DELTA_H,
            GaitFeatureTypes.STEP_LENGTH_M1,
            GaitFeatureTypes.STEP_LENGTH_M1_ASYMMETRY,
            GaitFeatureTypes.STRIDE_LENGTH_M1,
            GaitFeatureTypes.STRIDE_LENGTH_M1_ASYMMETRY,
            GaitFeatureTypes.GAIT_SPEED_M1,
            GaitFeatureTypes.GAIT_SPEED_M1_ASYMMETRY,
            GaitFeatureTypes.INTRA_STEP_COVARIANCE_V,
            GaitFeatureTypes.INTRA_STRIDE_COVARIANCE_V,
            GaitFeatureTypes.HARMONIC_RATIO_V,
            GaitFeatureTypes.STRIDE_SPARC,
        ]
        self.bout_gait_metrics: List[GaitFeatureTypes] = [
            GaitFeatureTypes.BOUT_DURATION,
            GaitFeatureTypes.BOUT_STEPS,
            GaitFeatureTypes.GAIT_CYCLES,
            GaitFeatureTypes.DEBUG_MEAN_STEP_FREQ,
            GaitFeatureTypes.BOUT_PHASE_COORDINATION_INDEX,
            GaitFeatureTypes.BOUT_GAIT_SYMMETRY_INDEX,
            GaitFeatureTypes.BOUT_STEP_REGULARITY_V,
            GaitFeatureTypes.BOUT_STRIDE_REGULARITY_V,
            GaitFeatureTypes.BOUT_AUTOCOVARIANCE_SYMMETRY_V,
            GaitFeatureTypes.BOUT_REGULARITY_INDEX_V,
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
        print(
            self.get_unique_bouts_count(
                gait_res, 0, len(gait_res[GaitFeatureTypes.DAY_N.value])
            )
        )
        multi_day_metrics: List[Dict[GaitFeatureTypes, float]] = (
            self.aggregate_multi_day_metrics(gait_res)
        )
        output_features = self.compute_aggregate_features(multi_day_metrics)

        # Example: Visualize multiple gait parameters
        parameters_to_plot = [
            GaitFeatureTypes.GAIT_SPEED,
            GaitFeatureTypes.STRIDE_LENGTH,
            GaitFeatureTypes.CADENCE,
            GaitFeatureTypes.STRIDE_TIME,
        ]
        self.visualize_gait_parameters(output_features, parameters_to_plot, n_cols=2)

        print("")

    def compute_aggregate_features(
        self, multi_day_metrics: List[Dict[GaitFeatureTypes, float]]
    ) -> List[AggregateFeatureSetEntry]:
        raw_feature_types: List[GaitFeatureTypes] = multi_day_metrics[0].keys()
        output_features = {}
        for raw_feature_type in raw_feature_types:
            feature_values = np.array(
                [features[raw_feature_type] for features in multi_day_metrics]
            )
            aggregate_features: Dict[DescriptiveStatisticsTypes, float] = {}
            for type in DescriptiveStatisticsTypes:
                aggregate_features[type] = type(feature_values)
            output_features[raw_feature_type] = aggregate_features
        return output_features

    def aggregate_multi_day_metrics(
        self, gait_res: Dict
    ) -> List[Dict[GaitFeatureTypes, float]]:
        """Aggregates multi-day, event-level metrics into collection of bout-level metrics

        Args:
            gait_res (Dict): _description_

        Returns:
            List[Dict[GaitFeatureKeys, float]]: _description_
        """
        # Init results dictionary (dayN - boutN)
        multi_day_metrics: List[Dict[GaitFeatureTypes, float]] = []
        # Initialize pointers for to traverse days and bouts
        day_start_ix = 0
        day_n = 1
        # Reference bout numbers and days from resutls
        day_n_list = gait_res[GaitFeatureTypes.DAY_N.value]
        # Traverse days
        while day_start_ix < len(day_n_list):
            # [1, 1, 1, 2, 2, 3]
            # print(f"Day number: {day_n}")
            day_end_ix = day_start_ix
            # Traverse days to find end day index
            while day_end_ix < len(day_n_list) and day_n_list[day_end_ix] == day_n:
                day_end_ix += 1
            # Aggregate bouts, add to result
            bout_metrics = self.aggregate_single_day_metrics(
                gait_res, day_start_ix, day_end_ix
            )
            multi_day_metrics.extend(bout_metrics)
            # print(
            #     f"Bout count: {self.get_unique_bouts_count(gait_res, day_start_ix, day_end_ix)}"
            # )
            # print(f"Bouts calc'd: {len(bout_metrics)}")
            # Increment day_n and day_start_ix
            day_n += 1
            day_start_ix = day_end_ix
        # Return results
        return multi_day_metrics

    def aggregate_single_day_metrics(
        self, gait_res, day_start_ix: int, day_end_ix: int
    ) -> List[Dict[GaitFeatureTypes, np.float64]]:
        single_day_metrics = []
        bout_n_list = gait_res[GaitFeatureTypes.BOUT_N.value]
        bout_start_ix = day_start_ix
        bout_n = 1
        while bout_start_ix < len(bout_n_list) and bout_start_ix < day_end_ix:
            bout_metrics = {}
            bout_end_ix = bout_start_ix
            while bout_end_ix < day_end_ix and bout_n_list[bout_end_ix] == bout_n:
                bout_end_ix += 1
            # print(f"BOUT N: {bout_n}")
            # print(f"BOUT START: {bout_start_ix}")
            # print(f"BOUT END: {bout_end_ix}")

            for event_metric in self.event_gait_metrics:
                # Take mean of metrics ignoring nan values
                bout_metrics[event_metric] = np.nanmean(
                    gait_res[event_metric.value][bout_start_ix:bout_end_ix]
                )
            for bout_metric in self.bout_gait_metrics:
                bout_metrics[bout_metric] = np.float64(
                    gait_res[bout_metric.value][bout_start_ix]
                )
            bout_metrics[GaitFeatureTypes.BOUT_START_TIMESTAMP] = gait_res[GaitFeatureTypes.IC_TIME.value][bout_start_ix]
            bout_metrics[GaitFeatureTypes.BOUT_END_TIMESTAMP] = gait_res[GaitFeatureTypes.IC_TIME.value][bout_end_ix]
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

    def get_unique_bouts_count(self, gait_res, start, stop):
        unique_bout_ids = []
        for i in range(start, stop):
            bout_id = (
                str(gait_res[GaitFeatureTypes.DAY_N.value][i])
                + "_"
                + str(gait_res[GaitFeatureTypes.BOUT_N.value][i])
            )
            unique_bout_ids.append(bout_id)
        return len(set(unique_bout_ids))

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

    def visualize_gait_parameters(
        self,
        output_features: Dict,
        gait_parameters: List[GaitFeatureTypes],
        n_cols: int = 2,
    ):
        """Creates multiple box and whisker plots for specified gait parameters.

        Args:
            output_features (Dict): Nested dictionary of gait parameters and their descriptive statistics
            gait_parameters (List[GaitFeatureTypes]): List of gait parameters to visualize
            n_cols (int, optional): Number of columns in the subplot grid. Defaults to 2.
        """
        # Calculate number of rows needed
        n_rows = (len(gait_parameters) + n_cols - 1) // n_cols

        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

        # Convert axes to 1D array if there's only one row
        if n_rows == 1:
            axes = np.array([axes])
        # Convert to 2D array if there's only one column
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Box plot statistics mapping
        box_plot_stats = {
            DescriptiveStatisticsTypes.MIN: "Min",
            DescriptiveStatisticsTypes.PERCENTILE_25: "Q1",
            DescriptiveStatisticsTypes.MEDIAN: "Median",
            DescriptiveStatisticsTypes.PERCENTILE_75: "Q3",
            DescriptiveStatisticsTypes.MAX: "Max",
        }

        # Create plots
        for idx, (ax, gait_parameter) in enumerate(zip(axes_flat, gait_parameters)):
            stats_dict = output_features[gait_parameter]

            # Collect data for plotting
            data = [stats_dict[stat_type] for stat_type in box_plot_stats.keys()]

            # Create box plot
            ax.boxplot([data], labels=[gait_parameter.value])

            # Add mean as a point if available
            if DescriptiveStatisticsTypes.MEAN in stats_dict:
                mean_value = stats_dict[DescriptiveStatisticsTypes.MEAN]
                ax.plot(1, mean_value, "ro", label="Mean")
                ax.legend()

            # Customize plot
            ax.set_title(f"Distribution of {gait_parameter.value}")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="x", rotation=45)

        # Hide any unused subplots
        for idx in range(len(gait_parameters), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Adjust layout
        plt.tight_layout()
        plt.show()


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
