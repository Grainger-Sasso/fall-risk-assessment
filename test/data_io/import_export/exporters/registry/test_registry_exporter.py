import os
import tempfile
from pathlib import Path

from src.data_io.import_export.exporters.registry.registry_exporter import (
    RegistryExporter,
)
from src.data_io.import_export.importers.registry.registry_importer import (
    RegistryFileNames,
    RegistryImporter,
)
from src.database_manager.registry.registry import Registry
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
)


class TestRegistryExporter(BaseTest):
    def setUp(self):
        self.exporter = RegistryExporter()
        self.importer = RegistryImporter()
        self.helper = DatabaseManagerTestHelper()

        # Create temp directory for test
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test data
        self.test_data = self.helper.create_test_registry()

    def tearDown(self):
        # Clean up test directory
        if self.temp_path.exists():
            # Walk directory tree bottom-up to properly remove all contents
            for root, dirs, files in os.walk(self.temp_path, topdown=False):
                # First remove all files in current directory
                for name in files:
                    (Path(root) / name).unlink()
                # Then remove the directory itself
                for name in dirs:
                    (Path(root) / name).rmdir()
            # Finally remove the temp directory
            self.temp_path.rmdir()

    def test_export_data(self):
        # Export the test data
        output_file_path: Path = self.exporter.export_data(
            self.temp_path, self.test_data
        )

        # Verify export succeeded with correct output path
        self.assertIsInstance(output_file_path, Path)
        expected_file = self.temp_path / f"{RegistryFileNames.REGISTRY.value}.csv"
        self.assertEqual(str(output_file_path), str(expected_file))

        # Verify file was created
        self.assertTrue(expected_file.exists(), "Output file not created")
        self.assertTrue(expected_file.is_file(), "Output is not a file")

        # Import the exported data
        result = self.importer.import_data(self.temp_path, type(TestSourceIdentifier))

        # Verify result matches test data
        self.assertEqual(len(result.registry), len(self.test_data.registry))
        for id_val, path in result.registry.items():
            self.assertIn(id_val, self.test_data.registry)
            self.assertEqual(path, self.test_data.registry[id_val])

    def test_export_data_file_exists(self):
        # Create test data with different registry entries
        first_data = self.helper.create_test_registry_with_single_entry(
            0
        )  # First test entry
        second_data = self.helper.create_test_registry_with_single_entry(
            1
        )  # Second test entry

        # Create initial file
        first_path = self.exporter.export_data(self.temp_path, first_data)

        # Export again to same location with different data
        second_path = self.exporter.export_data(self.temp_path, second_data)

        # Verify paths are the same
        self.assertEqual(first_path, second_path)

        # Verify no copy file remains
        copy_path = self.temp_path / f"{RegistryFileNames.REGISTRY.value}_copy.csv"
        self.assertFalse(copy_path.exists())

        # Import and verify the data matches the second export
        result = self.importer.import_data(self.temp_path, type(TestSourceIdentifier))
        self.assertEqual(len(result.registry), 1)
        self.assertEqual(
            result.registry[TestConstants.TEST_SOURCE_IDS.value[1]],
            TestConstants.TEST_PATHS.value[1],
        )

    def test_export_data_write_fails(self):
        # Create directory with read-only permissions
        readonly_dir = self.temp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        with self.assertRaises(Exception) as context:
            self.exporter.export_data(readonly_dir, self.test_data)

        self.assertIn(
            "Export failed: [Errno 13] Permission denied:", str(context.exception)
        )

        # Clean up - restore permissions to allow deletion
        readonly_dir.chmod(0o777)


if __name__ == "__main__":
    TestRegistryExporter.run_tests()
