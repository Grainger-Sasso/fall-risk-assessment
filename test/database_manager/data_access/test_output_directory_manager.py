import tempfile
from pathlib import Path

from src.database_manager.data_access.output_directory_manager import (
    OutputDirectoryManager,
)
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestOutputDirectoryManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()
        self.test_path = Path(tempfile.mkdtemp())
        self.manager = OutputDirectoryManager({TestSourceIdentifier: self.test_path})

    def test_get_path(self):
        # Test successful path retrieval
        source_id = TestSourceIdentifier(TestConstants.TEST_SOURCE_IDS.value[0])
        path = self.manager.get_provider(type(source_id))
        self.assertIsInstance(path, Path)
        self.assertEqual(path, self.test_path)

    def test_invalid_data_type(self):
        # Test nonexistent identifier
        with self.assertRaises(KeyError):
            self.manager.get_provider(type(TestTargetIdentifier("nonexistent")))

    def test_validation_rejects_none_path(self):
        with self.assertRaises(ValueError) as ctx:
            OutputDirectoryManager({TestSourceIdentifier: None})
        self.assertIn("is None", str(ctx.exception))

    def test_validation_rejects_nonexistent_path(self):
        with self.assertRaises(ValueError) as ctx:
            OutputDirectoryManager({TestSourceIdentifier: Path("/nonexistent/path/12345")})
        self.assertIn("does not exist", str(ctx.exception))

    def test_validation_rejects_file_path(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                OutputDirectoryManager({TestSourceIdentifier: file_path})
            self.assertIn("not a directory", str(ctx.exception))
        finally:
            file_path.unlink(missing_ok=True)

    def test_validation_rejects_non_path_type(self):
        with self.assertRaises(ValueError) as ctx:
            OutputDirectoryManager({TestSourceIdentifier: "/some/string/path"})
        self.assertIn("not a Path", str(ctx.exception))


if __name__ == "__main__":
    TestOutputDirectoryManager.run_tests()
