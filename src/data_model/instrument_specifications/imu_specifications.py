from dataclasses import dataclass, field
from typing import Dict, List

from src.data_model.assessment_data import AssessmentData
from src.data_model.instrument_specifications.sensor_specifications import (
    SensorSpecification,
)
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.identifier import Identifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


@dataclass
class IMUSpecifications(AssessmentData):
    """
    Represents the specifications of an IMU device.
    """

    sensor_specifications: List[SensorSpecification]
    specification_id: InstrumentSpecificationIdentifier
    imu_name: str
    _sensor_type_to_specification_map: Dict[SensorType, SensorSpecification] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        # Create the axis maps during initialization
        self._sensor_type_to_specification_map = {
            spec.sensor_type: spec for spec in self.sensor_specifications
        }

    def get_specification_by_sensor_type(self, sensor_type: SensorType):
        """
        Retrieve sensor spec by sensor type
        """
        if sensor_type not in self._sensor_type_to_specification_map:
            raise ValueError(f"{sensor_type} not present in data")
        return self._sensor_type_to_specification_map[sensor_type]

    def get_data_id(self) -> Identifier:
        return self.specification_id
    
    def get_associated_data_id(self) -> Identifier:
        return None
