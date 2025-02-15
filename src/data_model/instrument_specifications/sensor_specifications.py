from dataclasses import dataclass
from typing import Dict, Tuple

from data_types.instrument.sensor_type import SensorType


@dataclass
class SensorSpecification:
    """
    Specifications of an IMU sensor
    """

    # Sensor type
    sensor_type: SensorType
    # Sensor name
    sensor_name: str
    # Native units
    units: str
    # Operating range
    range: Tuple[float, float]
    # Sensitivity as linear change in output per change in input
    sensitivity: float
    # Resolution in bits
    resolution: int
    # Sampling rate in Hz
    sampling_rate: float
    # Noise density in (µg/√Hz)
    noise_density: float
    # Bias stability in (°/s)
    bias_stability: float
    # Alignment error in percent
    alignment_error: float
    # Cross axis sensitivity in percent
    cross_axis_sensitivity: float
    # Power consumption in mW
    power_consumption: float
    # Environmental operating conditions as condition: value
    operating_conditions: Dict[str, str]
    # Size as length, width, height in cm
    physical_size: Tuple[float, float, float]
    # Mass in g
    mass: float
