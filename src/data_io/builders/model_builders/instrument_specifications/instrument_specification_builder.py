from typing import Any, Dict, List, Tuple

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.model_fields.instrument_specification.instrument_specification_fields import (
    InstrumentSpecificationFields,
)
from src.data_model.instrument_specifications.imu_specifications import (
    IMUSpecifications,
)
from src.data_model.instrument_specifications.sensor_specifications import (
    SensorSpecification,
)
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


class IMUSpecificationBuilder(ModelBuilder):
    """IMU specification builder"""

    version = "1.0"

    def build(self, input_file: JSONDictFile) -> IMUSpecifications:
        if not isinstance(input_file, JSONDictFile):
            raise ValueError("Invalid imu spec JSON file")
        return self.__build_imu_spec(input_file)

    def __build_imu_spec(self, input_file: JSONDictFile) -> IMUSpecifications:
        # Get spec ID
        spec_id: InstrumentSpecificationIdentifier = InstrumentSpecificationIdentifier(
            input_file.data[InstrumentSpecificationFields.SPEC_ID.value]
        )
        # Get IMU name
        imu_name: str = input_file.data[InstrumentSpecificationFields.IMU_NAME.value]
        # Build sensor specification list
        file_sensor_specifications: List[Dict[str, Any]] = input_file.data[
            InstrumentSpecificationFields.SENSOR_SPECIFICATIONS.value
        ]
        sensor_specification_list: List[SensorSpecification] = (
            self.__build_sensor_specification_list(file_sensor_specifications)
        )
        return IMUSpecifications(sensor_specification_list, spec_id, imu_name)

    def __build_sensor_specification_list(
        self, file_sensor_specifications: List[Dict[str, Any]]
    ) -> List[SensorSpecification]:
        sensor_spec_list: List[SensorSpecification] = []
        for file_sensor_spec in file_sensor_specifications:
            sensor_spec_list.append(self.__build_sensor_spec(file_sensor_spec))
        return sensor_spec_list

    def __build_sensor_spec(
        self, file_sensor_spec: Dict[str, Any]
    ) -> SensorSpecification:
        sensor_type: SensorType = SensorType(
            file_sensor_spec[InstrumentSpecificationFields.SENSOR_TYPE.value]
        )
        sensor_name: str = file_sensor_spec[
            InstrumentSpecificationFields.SENSOR_NAME.value
        ]
        units: str = file_sensor_spec[InstrumentSpecificationFields.UNITS.value]
        range: Tuple[float, float] = (file_sensor_spec[
            InstrumentSpecificationFields.RANGE.value
        ][InstrumentSpecificationFields.VALUE.value])
        sensitivity: float = file_sensor_spec[
            InstrumentSpecificationFields.SENSITIVITY.value
        ][InstrumentSpecificationFields.VALUE.value]
        resolution: int = file_sensor_spec[
            InstrumentSpecificationFields.RESOLUTION.value
        ][InstrumentSpecificationFields.VALUE.value]
        sampling_rate: float = file_sensor_spec[
            InstrumentSpecificationFields.SAMPLING_RATE.value
        ][InstrumentSpecificationFields.VALUE.value]
        noise_density: float = file_sensor_spec[
            InstrumentSpecificationFields.NOISE_DENSITY.value
        ][InstrumentSpecificationFields.VALUE.value]
        bias_stability: float = file_sensor_spec[
            InstrumentSpecificationFields.BIAS_STABILITY.value
        ][InstrumentSpecificationFields.VALUE.value]
        alignment_error: float = file_sensor_spec[
            InstrumentSpecificationFields.ALIGNMENT_ERROR.value
        ][InstrumentSpecificationFields.VALUE.value]
        cross_axis_sensitivity: float = file_sensor_spec[
            InstrumentSpecificationFields.CROSS_AXIS_SENSITIVITY.value
        ][InstrumentSpecificationFields.VALUE.value]
        power_consumption: float = file_sensor_spec[
            InstrumentSpecificationFields.POWER_CONSUMPTION.value
        ][InstrumentSpecificationFields.VALUE.value]
        operating_conditions: Dict[str, str] = file_sensor_spec[
            InstrumentSpecificationFields.OPERATING_CONDITIONS.value
        ][InstrumentSpecificationFields.VALUE.value]
        size: Dict[str, float] = file_sensor_spec[
            InstrumentSpecificationFields.PHYSICAL_SIZE.value
        ]
        physical_size: Tuple[float, float, float] = tuple(
            [
                size[InstrumentSpecificationFields.LENGTH.value],
                size[InstrumentSpecificationFields.WIDTH.value],
                size[InstrumentSpecificationFields.HEIGHT.value]
            ]
        )
        mass: float = file_sensor_spec[InstrumentSpecificationFields.MASS.value]
        return SensorSpecification(
            sensor_type,
            sensor_name,
            units,
            range,
            sensitivity,
            resolution,
            sampling_rate,
            noise_density,
            bias_stability,
            alignment_error,
            cross_axis_sensitivity,
            power_consumption,
            operating_conditions,
            physical_size,
            mass,
        )
