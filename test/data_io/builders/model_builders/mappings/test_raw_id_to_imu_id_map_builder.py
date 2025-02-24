from src.data_io.builders.model_builders.mappings.raw_id_to_imu_id_map_builder import (
    RawFeatureIDToIMUIDMapBuilder,
)
from src.database_manager.mappings.raw_feature_id_to_imu_data_id_map import (
    RawFeatureIDToIMUDataIDMap,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import MappingHelper, TestConstants


class TestRawFeatureIDToIMUIDMapBuilder(BaseTest):
    def setUp(self):
        self.builder = RawFeatureIDToIMUIDMapBuilder()
        self.data_helper = MappingHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_mapping_csv(
            TestConstants.RAW_TO_IMU_SOURCE_IDS.value,
            TestConstants.RAW_TO_IMU_TARGET_IDS.value,
        )

        # Test building map
        result = self.builder.build(csv_data)

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

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestRawFeatureIDToIMUIDMapBuilder.run_tests()
