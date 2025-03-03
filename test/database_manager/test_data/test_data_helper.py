from enum import Enum
from pathlib import Path
from typing import Dict, List

from src.identifiers.identifier import Identifier


class TestConstants(Enum):
    """Constants for database manager tests"""

    # Test identifiers
    TEST_SOURCE_IDS = ["source_1", "source_2", "source_3"]
    TEST_TARGET_IDS = ["target_1", "target_2", "target_3"]

    # Test paths
    TEST_PATHS = ["/test/path/1", "/test/path/2", "/test/path/3"]


class TestIdentifier(Identifier):
    """Test implementation of Identifier for testing"""

    def __init__(self, value: str):
        super().__init__(value)

    def validate(self, value: str) -> bool:
        return True


class DatabaseManagerTestHelper:
    """Helper class for creating test data for database manager tests"""

    @staticmethod
    def create_test_mapping(
        source_ids: List[str], target_ids: List[str]
    ) -> Dict[TestIdentifier, TestIdentifier]:
        """Create test mapping between identifiers

        Args:
            source_ids: List of source identifier values
            target_ids: List of target identifier values

        Returns:
            Dictionary mapping source to target identifiers
        """
        return {
            TestIdentifier(src): TestIdentifier(tgt)
            for src, tgt in zip(source_ids, target_ids)
        }

    @staticmethod
    def create_test_registry(
        ids: List[str], paths: List[str]
    ) -> Dict[TestIdentifier, Path]:
        """Create test registry mapping identifiers to paths

        Args:
            ids: List of identifier values
            paths: List of path strings

        Returns:
            Dictionary mapping identifiers to paths
        """
        return {TestIdentifier(id): Path(path) for id, path in zip(ids, paths)}
