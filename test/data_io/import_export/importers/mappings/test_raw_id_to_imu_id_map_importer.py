import tempfile
from pathlib import Path

from src.data_io.import_export.importers.mappings.raw_id_to_imu_id_map_importer import (
    RawIDToIMUIDMapFileNames,
    RawIDToIMUIDMapImporter,
)
from src.database_manager.mappings.raw_feature_id_to_imu_data_id_map import (
    RawFeatureIDToIMUDataIDMap,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import MappingHelper, TestConstants


class TestRawIDToIMUIDMapImporter(BaseTest):
    def setUp(self):
        self.importer = RawIDToIMUIDMapImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = MappingHelper()

        # Create test file in temp directory
        self.map_path = self.helper.create_test_mapping_file(
            TestConstants.RAW_TO_IMU_SOURCE_IDS.value,
            TestConstants.RAW_TO_IMU_TARGET_IDS.value,
            self.temp_path
            / f"{RawIDToIMUIDMapFileNames.RAW_FEATURE_TO_IMU_DATA_ID_MAP.value}.csv",
        )

    def tearDown(self):
        # Clean up test files
        if self.map_path.exists():
            self.map_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Assertions
        self.assertIsInstance(result, RawFeatureIDToIMUDataIDMap)
        self.assertEqual(
            len(result.map), len(TestConstants.RAW_TO_IMU_SOURCE_IDS.value)
        )

        # Test mappings
        for ix, (raw_id, imu_id) in enumerate(result.map.items()):
            self.assertIsInstance(raw_id, RawFeatureIdentifier)
            self.assertIsInstance(imu_id, IMUDataIdentifier)
            self.assertEqual(
                raw_id.value, TestConstants.RAW_TO_IMU_SOURCE_IDS.value[ix]
            )
            self.assertEqual(
                imu_id.value, TestConstants.RAW_TO_IMU_TARGET_IDS.value[ix]
            )

    def test_missing_file(self):
        # Remove the required file
        self.map_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestRawIDToIMUIDMapImporter.run_tests()
