from typing import Any, Dict, List

import numpy as np

from src.data_model.data.imu.sensor_data import SensorData
from src.gait_features.contracts.extraction_result import BoutSegment


def to_seconds(time_values: np.ndarray) -> np.ndarray:
    if np.issubdtype(time_values.dtype, np.datetime64):
        return time_values.astype("datetime64[ns]").astype(np.int64) / 1e9
    try:
        return time_values.astype(float)
    except Exception:
        return np.array([float(item.timestamp()) for item in time_values], dtype=float)


def full_recording_bout(sensor_data: SensorData) -> BoutSegment:
    numeric_time = np.asarray(sensor_data.time, dtype=float)
    if numeric_time.size == 0:
        raise ValueError("Sensor time series is empty.")
    start_time, end_time = float(numeric_time[0]), float(numeric_time[-1])
    if end_time <= start_time:
        end_time = start_time + 1e-6
    return BoutSegment(
        sample_start=0,
        sample_end=numeric_time.size,
        event_start=0,
        event_end=0,
        start_time=start_time,
        end_time=end_time,
    )


def resolve_skdh_bout_segments(
    gait_data: Dict[str, Any],
    sensor_data: SensorData,
    treadmill_profile: bool,
) -> List[BoutSegment]:
    day_values = gait_data.get("Day N")
    bout_values = gait_data.get("Bout N")
    ic_time_values = gait_data.get("IC Time")
    numeric_time = np.asarray(sensor_data.time, dtype=float)
    if day_values is None or bout_values is None or ic_time_values is None:
        return [full_recording_bout(sensor_data)]
    day_array = np.asarray(day_values)
    bout_array = np.asarray(bout_values)
    ic_seconds = to_seconds(np.asarray(ic_time_values))
    if day_array.size == 0 or bout_array.size == 0 or ic_seconds.size == 0:
        return [full_recording_bout(sensor_data)]

    length = min(day_array.size, bout_array.size, ic_seconds.size)
    day_array, bout_array, ic_seconds = day_array[:length], bout_array[:length], ic_seconds[:length]
    ranges: List[BoutSegment] = []
    start_index = 0
    while start_index < length:
        end_index = start_index + 1
        while (
            end_index < length
            and day_array[end_index] == day_array[start_index]
            and bout_array[end_index] == bout_array[start_index]
        ):
            end_index += 1
        start_time = float(ic_seconds[start_index])
        end_time = float(ic_seconds[end_index - 1])
        if end_time <= start_time:
            end_time = start_time + 1e-6
        sample_start = int(np.searchsorted(numeric_time, start_time, side="left"))
        sample_end = int(np.searchsorted(numeric_time, end_time, side="right"))
        sample_start = max(0, min(sample_start, numeric_time.size - 1))
        sample_end = max(sample_start + 1, min(sample_end, numeric_time.size))
        ranges.append(
            BoutSegment(
                sample_start=sample_start,
                sample_end=sample_end,
                event_start=start_index,
                event_end=end_index,
                start_time=start_time,
                end_time=end_time,
            )
        )
        start_index = end_index
    if ranges:
        return ranges
    return [full_recording_bout(sensor_data)]
