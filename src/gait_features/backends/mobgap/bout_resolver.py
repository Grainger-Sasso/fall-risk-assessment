from typing import Any, List

import numpy as np

from src.data_model.data.imu.sensor_data import SensorData
from src.gait_features.contracts.extraction_result import BoutSegment


def resolve_mobgap_bout_segments(
    per_wb_parameters: Any,
    sensor_data: SensorData,
) -> List[BoutSegment]:
    if per_wb_parameters is None or len(per_wb_parameters) == 0:
        numeric_time = np.asarray(sensor_data.time, dtype=float)
        if numeric_time.size == 0:
            raise ValueError("Sensor time series is empty.")
        return [
            BoutSegment(
                sample_start=0,
                sample_end=numeric_time.size,
                event_start=0,
                event_end=0,
                start_time=float(numeric_time[0]),
                end_time=float(numeric_time[-1]),
            )
        ]

    numeric_time = np.asarray(sensor_data.time, dtype=float)
    segments: List[BoutSegment] = []
    for event_index, (_, row) in enumerate(per_wb_parameters.iterrows()):
        sample_start = int(row["start"])
        sample_end = int(row["end"])
        sample_start = max(0, min(sample_start, numeric_time.size - 1))
        sample_end = max(sample_start + 1, min(sample_end, numeric_time.size))
        start_time = float(numeric_time[sample_start])
        end_time = float(numeric_time[min(sample_end - 1, numeric_time.size - 1)])
        if end_time <= start_time:
            end_time = start_time + 1e-6
        segments.append(
            BoutSegment(
                sample_start=sample_start,
                sample_end=sample_end,
                event_start=event_index,
                event_end=event_index + 1,
                start_time=start_time,
                end_time=end_time,
            )
        )
    return segments
