from dataclasses import dataclass

from src.data_types.instrument.sensor_type import SensorType


@dataclass
class SensorMetadata:
    """
    Metadata for SensorData
    """

    sensor_type: SensorType
    sampling_rate: float
    unit: str
