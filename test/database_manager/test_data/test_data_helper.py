from enum import Enum
from pathlib import Path

from src.data_model.mapping.mapping import Mapping
from src.data_model.registry.registry import Registry
from src.identifiers.identifier import Identifier


class TestConstants(Enum):
    """Constants for database manager tests"""

    # Test identifiers
    TEST_SOURCE_IDS = ["source_1", "source_2", "source_3"]
    TEST_TARGET_IDS = ["target_1", "target_2", "target_3"]

    # Test paths
    TEST_PATHS = [Path("/test/path/1"), Path("/test/path/2"), Path("/test/path/3")]


class TestSourceIdentifier(Identifier):
    """Test implementation of Identifier for testing"""

    def __init__(self, value: str):
        super().__init__(value)

    def validate(self, value: str) -> bool:
        return True


class TestTargetIdentifier(Identifier):
    """Test implementation of Identifier for testing"""

    def __init__(self, value: str):
        super().__init__(value)

    def validate(self, value: str) -> bool:
        return True


class DatabaseManagerTestHelper:
    """Helper class for creating test data for database manager tests"""

    def create_test_identifier(self) -> TestSourceIdentifier:
        return TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])

    def create_test_mapping(self) -> Mapping:
        """Create test mapping between identifiers

        Returns:
            Dictionary mapping source to target identifiers
        """
        return Mapping(
            {
                src: tgt
                for src, tgt in zip(
                    TestConstants.TEST_SOURCE_IDS.value,
                    TestConstants.TEST_TARGET_IDS.value,
                )
            },
            source_id_type=TestSourceIdentifier,
            target_id_type=TestTargetIdentifier,
            subdir_path=TestConstants.TEST_PATHS.value[0],
        )

    def create_test_registry(self) -> Registry:
        """Create test registry mapping identifiers to paths

        Returns:
            Dictionary mapping identifiers to paths
        """
        return Registry(
            {
                id: Path(path)
                for id, path in zip(
                    TestConstants.TEST_SOURCE_IDS.value[:2],
                    TestConstants.TEST_PATHS.value[:2],
                )
            },
            id_type=TestSourceIdentifier,
            subdir_path=TestConstants.TEST_PATHS.value[0],
        )

    def create_test_mapping_with_single_entry(self, index: int) -> Mapping:
        """Create test mapping with single entry at given index"""
        return Mapping(
            {
                TestConstants.TEST_SOURCE_IDS.value[index]: 
                TestConstants.TEST_TARGET_IDS.value[index]
            },
            source_id_type=TestSourceIdentifier,
            target_id_type=TestTargetIdentifier,
            subdir_path=TestConstants.TEST_PATHS.value[0],
        )

    def create_test_registry_with_single_entry(self, index: int) -> Registry:
        """Create test registry with single entry at given index"""
        return Registry(
            {
                TestConstants.TEST_SOURCE_IDS.value[index]: 
                TestConstants.TEST_PATHS.value[index]
            },
            id_type=TestSourceIdentifier,
            subdir_path=TestConstants.TEST_PATHS.value[0],
        )
