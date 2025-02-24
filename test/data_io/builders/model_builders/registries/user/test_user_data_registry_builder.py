from pathlib import Path
from src.data_io.builders.model_builders.registries.user.user_data_registry_builder import (
    UserDataRegistryBuilder,
)
from src.database_manager.registries.user.user_data_registry import UserDataRegistry
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import RegistryHelper, TestConstants


class TestUserDataRegistryBuilder(BaseTest):
    def setUp(self):
        self.builder = UserDataRegistryBuilder()
        self.data_helper = RegistryHelper()

    def test_build_valid_data(self):
        # Create test CSV data
        csv_data = self.data_helper.create_test_registry_csv(
            TestConstants.USER_REGISTRY_IDS.value
        )

        # Test building registry
        result = self.builder.build(csv_data)

        # Assertions
        self.assertIsInstance(result, UserDataRegistry)
        self.assertEqual(
            len(result.registry), len(TestConstants.USER_REGISTRY_IDS.value)
        )

        # Test registry entries
        for ix, (user_id, path) in enumerate(result.registry.items()):
            self.assertIsInstance(user_id, UserIdentifier)
            self.assertIsInstance(path, Path)
            self.assertEqual(
                user_id.value, TestConstants.USER_REGISTRY_IDS.value[ix]
            )
            self.assertEqual(
                str(path), TestConstants.REGISTRY_PATHS.value[ix]
            )

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None)


if __name__ == "__main__":
    TestUserDataRegistryBuilder.run_tests() 