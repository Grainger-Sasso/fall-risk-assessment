import tempfile
from pathlib import Path

from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationFileNames,
    InstrumentSpecificationImporter,
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
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import (
    InstrumentSpecificationHelper,
    TestConstants,
)


class TestInstrumentSpecificationImporter(BaseTest):
    def setUp(self):
        self.importer = InstrumentSpecificationImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = InstrumentSpecificationHelper()

        # Create test file in temp directory
        self.spec_path = self.helper.create_test_specification_file(
            self.temp_path
            / f"{InstrumentSpecificationFileNames.INSTRUMENT_SPEC.value}.json"
        )

    def tearDown(self):
        # Clean up test files
        if self.spec_path.exists():
            self.spec_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Assertions
        self.assertIsInstance(result, IMUSpecifications)
        self.assertEqual(result.imu_name, TestConstants.IMU_NAME.value)
        self.assertIsInstance(
            result.specification_id, InstrumentSpecificationIdentifier
        )
        self.assertEqual(result.specification_id.value, TestConstants.SPEC_ID.value)

        # Test sensor specifications
        self.assertIsInstance(result.sensor_specifications, list)
        self.assertEqual(len(result.sensor_specifications), 2)

        for spec in result.sensor_specifications:
            self.assertIsInstance(spec, SensorSpecification)
            self.assertIn(
                spec.sensor_type, [SensorType.ACCELEROMETER, SensorType.GYROSCOPE]
            )
            self.assertEqual(spec.sensor_name, TestConstants.SENSOR_NAME.value)
            self.assertEqual(spec.units, TestConstants.SENSOR_UNITS.value)
            self.assertEqual(spec.range, TestConstants.SENSOR_RANGE.value)
            self.assertEqual(spec.sensitivity, TestConstants.SENSOR_SENSITIVITY.value)
            self.assertEqual(spec.resolution, TestConstants.SENSOR_RESOLUTION.value)
            self.assertEqual(
                spec.sampling_rate, TestConstants.SENSOR_SAMPLING_RATE.value
            )
            self.assertEqual(
                spec.noise_density, TestConstants.SENSOR_NOISE_DENSITY.value
            )
            self.assertEqual(
                spec.bias_stability, TestConstants.SENSOR_BIAS_STABILITY.value
            )
            self.assertEqual(
                spec.alignment_error, TestConstants.SENSOR_ALIGNMENT_ERROR.value
            )
            self.assertEqual(
                spec.cross_axis_sensitivity,
                TestConstants.SENSOR_CROSS_AXIS_SENSITIVITY.value,
            )
            self.assertEqual(
                spec.power_consumption, TestConstants.SENSOR_POWER_CONSUMPTION.value
            )
            self.assertEqual(
                spec.operating_conditions,
                TestConstants.SENSOR_OPERATING_CONDITIONS.value,
            )
            self.assertEqual(
                spec.physical_size,
                tuple(
                    [
                        TestConstants.SENSOR_SIZE.value["length"],
                        TestConstants.SENSOR_SIZE.value["width"],
                        TestConstants.SENSOR_SIZE.value["height"],
                    ]
                ),
            )
            self.assertEqual(spec.mass, TestConstants.SENSOR_MASS.value)

    def test_missing_file(self):
        # Remove the required file
        self.spec_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestInstrumentSpecificationImporter.run_tests()
