from enum import Enum
from pathlib import Path

from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier


class TestConstants(Enum):
    """Constants for database manager tests"""

    # Test identifiers
    TEST_SOURCE_IDS = ["source_1", "source_2", "source_3"]
    TEST_TARGET_IDS = ["target_1", "target_2", "target_3"]

    # Test paths
    TEST_PATHS = ["/test/path/1", "/test/path/2", "/test/path/3"]


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
        )
