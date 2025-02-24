from src.data_io.builders.model_builders.mappings.imu_id_to_user_id_map_builder import (
    IMUIDToUserIDMapBuilder,
)
from src.database_manager.mappings.imu_data_id_to_user_data_id_map import (
    IMUDataIDToUserDataIDMap,
)
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import MappingHelper, TestConstants


class TestIMUIDToUserIDMapBuilder(BaseTest):
    def setUp(self):
        self.builder = IMUIDToUserIDMapBuilder()
        self.data_helper = MappingHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_mapping_csv(
            TestConstants.IMU_TO_USER_SOURCE_IDS.value,
            TestConstants.IMU_TO_USER_TARGET_IDS.value,
        )

        # Test building map
        result = self.builder.build(csv_data)

        # Assertions
        self.assertIsInstance(result, IMUDataIDToUserDataIDMap)
        self.assertEqual(
            len(result.map), len(TestConstants.IMU_TO_USER_SOURCE_IDS.value)
        )

        # Test mappings
        for ix, (imu_id, user_id) in enumerate(result.map.items()):
            self.assertIsInstance(imu_id, IMUDataIdentifier)
            self.assertIsInstance(user_id, UserIdentifier)
            self.assertEqual(
                imu_id.value, TestConstants.IMU_TO_USER_SOURCE_IDS.value[ix]
            )
            self.assertEqual(
                user_id.value, TestConstants.IMU_TO_USER_TARGET_IDS.value[ix]
            )

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestIMUIDToUserIDMapBuilder.run_tests()
