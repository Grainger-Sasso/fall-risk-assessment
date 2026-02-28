from pathlib import Path

from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.mapping.mapping import Mapping
from src.identifiers.identifier import Identifier
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class StrictValidationIdentifier(Identifier):
    """Identifier that rejects empty or 'invalid' values for validation tests."""

    def validate(self, value: str) -> bool:
        return bool(value and value.strip() and "invalid" not in value.lower())


class TestMappingManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()

        # Create test mapping using helper
        self.test_mapping = self.helper.create_test_mapping()
        self.manager = MappingManager({TestSourceIdentifier: self.test_mapping})

    def test_get_mapping(self):
        # Test successful mapping
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        mapping = self.manager.get_provider(type(source_id))
        self.assertIsInstance(mapping, Mapping)
        self.assertEqual(mapping, self.test_mapping)

    def test_invalid_id(self):
        # Test nonexistent mapping
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))

    def test_validation_rejects_empty_source_id(self):
        mapping = Mapping(
            {"": "target_1"},
            source_id_type=TestSourceIdentifier,
            target_id_type=TestTargetIdentifier,
            subdir_path=Path("."),
        )
        with self.assertRaises(ValueError) as ctx:
            MappingManager({TestSourceIdentifier: mapping})
        self.assertIn("null or empty source id", str(ctx.exception))

    def test_validation_rejects_empty_target_id(self):
        mapping = Mapping(
            {"source_1": ""},
            source_id_type=TestSourceIdentifier,
            target_id_type=TestTargetIdentifier,
            subdir_path=Path("."),
        )
        with self.assertRaises(ValueError) as ctx:
            MappingManager({TestSourceIdentifier: mapping})
        self.assertIn("null or empty target id", str(ctx.exception))

    def test_validation_rejects_provider_key_mismatch(self):
        mapping = self.helper.create_test_mapping()
        with self.assertRaises(ValueError) as ctx:
            MappingManager({TestTargetIdentifier: mapping})
        self.assertIn("does not match", str(ctx.exception))

    def test_validation_rejects_invalid_source_id_format(self):
        mapping = Mapping(
            {"invalid_value": "target_1"},
            source_id_type=StrictValidationIdentifier,
            target_id_type=TestTargetIdentifier,
            subdir_path=Path("."),
        )
        with self.assertRaises(ValueError) as ctx:
            MappingManager({StrictValidationIdentifier: mapping})
        self.assertIn("invalid source id", str(ctx.exception))

    def test_validation_rejects_invalid_target_id_format(self):
        mapping = Mapping(
            {"source_1": "invalid_value"},
            source_id_type=TestSourceIdentifier,
            target_id_type=StrictValidationIdentifier,
            subdir_path=Path("."),
        )
        with self.assertRaises(ValueError) as ctx:
            MappingManager({TestSourceIdentifier: mapping})
        self.assertIn("invalid target id", str(ctx.exception))


if __name__ == "__main__":
    TestMappingManager.run_tests()
