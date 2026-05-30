from typing import Any, Dict

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.model_fields.instrument_specification.instrument_specification_fields import (
    InstrumentSpecificationFields,
)
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.data_model.instrument_specifications.sensor_specifications import SensorSpecification


class InstrumentSpecificationFileBuilder(FileBuilder):
    """Build JSON instrument specification files from IMUSpecifications models."""

    version = "1.0"

    def build(self, data: IMUSpecifications) -> JSONDictFile:
        if not isinstance(data, IMUSpecifications):
            raise ValueError("Input data must be IMUSpecifications")
        return JSONDictFile(
            data={
                InstrumentSpecificationFields.SPEC_ID.value: data.specification_id.value,
                InstrumentSpecificationFields.IMU_NAME.value: data.imu_name,
                InstrumentSpecificationFields.SENSOR_SPECIFICATIONS.value: [
                    self._build_sensor_spec_dict(sensor_spec)
                    for sensor_spec in data.sensor_specifications
                ],
            }
        )

    def _build_sensor_spec_dict(self, sensor_spec: SensorSpecification) -> Dict[str, Any]:
        return {
            InstrumentSpecificationFields.SENSOR_TYPE.value: sensor_spec.sensor_type.value,
            InstrumentSpecificationFields.SENSOR_NAME.value: sensor_spec.sensor_name,
            InstrumentSpecificationFields.UNITS.value: sensor_spec.units,
            InstrumentSpecificationFields.RANGE.value: {
                InstrumentSpecificationFields.VALUE.value: list(sensor_spec.range)
            },
            InstrumentSpecificationFields.SENSITIVITY.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.sensitivity
            },
            InstrumentSpecificationFields.RESOLUTION.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.resolution
            },
            InstrumentSpecificationFields.SAMPLING_RATE.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.sampling_rate
            },
            InstrumentSpecificationFields.NOISE_DENSITY.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.noise_density
            },
            InstrumentSpecificationFields.BIAS_STABILITY.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.bias_stability
            },
            InstrumentSpecificationFields.ALIGNMENT_ERROR.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.alignment_error
            },
            InstrumentSpecificationFields.CROSS_AXIS_SENSITIVITY.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.cross_axis_sensitivity
            },
            InstrumentSpecificationFields.POWER_CONSUMPTION.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.power_consumption
            },
            InstrumentSpecificationFields.OPERATING_CONDITIONS.value: {
                InstrumentSpecificationFields.VALUE.value: sensor_spec.operating_conditions
            },
            InstrumentSpecificationFields.PHYSICAL_SIZE.value: {
                InstrumentSpecificationFields.LENGTH.value: sensor_spec.physical_size[0],
                InstrumentSpecificationFields.WIDTH.value: sensor_spec.physical_size[1],
                InstrumentSpecificationFields.HEIGHT.value: sensor_spec.physical_size[2],
            },
            InstrumentSpecificationFields.MASS.value: sensor_spec.mass,
        }
