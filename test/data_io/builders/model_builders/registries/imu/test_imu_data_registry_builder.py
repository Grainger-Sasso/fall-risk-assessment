from pathlib import Path

from src.data_io.builders.model_builders.registries.imu.imu_data_registry_builder import (
    IMUDataRegistryBuilder,
)
from src.database_manager.registries.imu.imu_data_registry import IMUDataRegistry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestIMUDataRegistryBuilder(BaseTest):
    def setUp(self):
        self.builder = IMUDataRegistryBuilder()
        self.helper = RegistryHelper()

    def test_build(self):
        # Create test data
        test_data = self.helper.create_test_registry_csv(
            TestConstants.IMU_REGISTRY_IDS.value
        )

        # Build registry
        result = self.builder.build(test_data)

        # Verify result type
        self.assertIsInstance(result, IMUDataRegistry)

        # Verify registry contents
        for id, path in zip(
            TestConstants.IMU_REGISTRY_IDS.value,
            TestConstants.REGISTRY_PATHS.value,
        ):
            self.assertIn(IMUDataIdentifier(id), result.registry)
            self.assertEqual(result.registry[IMUDataIdentifier(id)], Path(path))


if __name__ == "__main__":
    TestIMUDataRegistryBuilder.run_tests() 