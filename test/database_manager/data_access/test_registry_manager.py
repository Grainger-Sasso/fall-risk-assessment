import tempfile
from pathlib import Path
from typing import Dict, Type

from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


def _create_registry_with_valid_paths() -> Registry:
    """Create a registry with temp directories that exist and contain files."""
    dir1 = tempfile.mkdtemp()
    dir2 = tempfile.mkdtemp()
    (Path(dir1) / "placeholder.txt").touch()
    (Path(dir2) / "placeholder.txt").touch()
    return Registry(
        {
            TestConstants.TEST_SOURCE_IDS.value[0]: Path(dir1),
            TestConstants.TEST_SOURCE_IDS.value[1]: Path(dir2),
        },
        id_type=TestSourceIdentifier,
        subdir_path=Path(dir1),
    )


class TestRegistryManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()
        self.registry = _create_registry_with_valid_paths()
        registries: Dict[Type[Identifier], Registry] = {
            self.registry.id_type: self.registry
        }
        self.manager = RegistryManager(registries)

    def test_get_registry(self):
        # Test successful registry retrieval
        test_id: TestSourceIdentifier = self.helper.create_test_identifier()
        registry = self.manager.get_provider(type(test_id))
        self.assertEqual(registry, self.registry)

    def test_invalid_id_type(self):
        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))

    def test_validation_rejects_empty_id(self):
        dir_path = Path(tempfile.mkdtemp())
        (dir_path / "placeholder.txt").touch()
        try:
            registry = Registry(
                {"": dir_path},
                id_type=TestSourceIdentifier,
                subdir_path=dir_path,
            )
            with self.assertRaises(ValueError) as ctx:
                RegistryManager({TestSourceIdentifier: registry})
            self.assertIn("null or empty id", str(ctx.exception))
        finally:
            (dir_path / "placeholder.txt").unlink()
            dir_path.rmdir()

    def test_validation_rejects_nonexistent_path(self):
        registry = Registry(
            {"source_1": Path("/nonexistent/path/12345")},
            id_type=TestSourceIdentifier,
            subdir_path=Path("."),
        )
        with self.assertRaises(ValueError) as ctx:
            RegistryManager({TestSourceIdentifier: registry})
        self.assertIn("do not exist", str(ctx.exception))

    def test_validation_rejects_file_path(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = Path(f.name)
        try:
            registry = Registry(
                {"source_1": file_path},
                id_type=TestSourceIdentifier,
                subdir_path=Path("."),
            )
            with self.assertRaises(ValueError) as ctx:
                RegistryManager({TestSourceIdentifier: registry})
            self.assertIn("not a directory", str(ctx.exception))
        finally:
            file_path.unlink(missing_ok=True)

    def test_validation_rejects_empty_directory(self):
        empty_dir = Path(tempfile.mkdtemp())
        try:
            registry = Registry(
                {"source_1": empty_dir},
                id_type=TestSourceIdentifier,
                subdir_path=Path("."),
            )
            with self.assertRaises(ValueError) as ctx:
                RegistryManager({TestSourceIdentifier: registry})
            self.assertIn("empty directory", str(ctx.exception))
        finally:
            empty_dir.rmdir()


if __name__ == "__main__":
    TestRegistryManager.run_tests()
