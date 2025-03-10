from pathlib import Path
from typing import Any, Dict, Type
from unittest.mock import MagicMock, patch

from src.data_io.import_export.importers.importer import Importer
from src.database_manager.data_access.export_manager import ExportManager
from src.database_manager.data_access.import_manager import ImportManager
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.identifier import Identifier
from test.base_test import BaseTest
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestDatabaseManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()

        # Create test registry
        self.registry = self.helper.create_test_registry()
        self.registry_manager = RegistryManager({TestSourceIdentifier: self.registry})

        # Create test mapping
        self.mapping = self.helper.create_test_mapping()
        self.mapping_manager = MappingManager({TestSourceIdentifier: self.mapping})

        # Create mock importer
        self.mock_importer = MagicMock(spec=Importer)
        self.mock_importer.import_data.return_value = "test_data"

        # Create import manager with mock importer
        self.import_manager = ImportManager({TestSourceIdentifier: self.mock_importer})

        # Create mock export manager
        self.export_manager = MagicMock(spec=ExportManager)

        # Create database manager
        self.db_manager = DatabaseManager(
            registry_manager=self.registry_manager,
            mapping_manager=self.mapping_manager,
            import_manager=self.import_manager,
            export_manager=self.export_manager,
        )

    def test_get_data(self):
        # Create test identifier
        test_id = self.helper.create_test_identifier()

        # Get data using database manager
        data = self.db_manager.import_data(test_id)

        # Verify correct path was retrieved from registry
        expected_path = self.registry.get_path(test_id)

        # Verify importer was called with correct path
        self.mock_importer.import_data.assert_called_once_with(expected_path)

        # Verify returned data matches mock importer output
        self.assertEqual(data, "test_data")

    def test_get_data_invalid_id_type(self):
        # Test with invalid identifier type
        invalid_id = TestTargetIdentifier("invalid")

        with self.assertRaises(KeyError):
            self.db_manager.import_data(invalid_id)

    def test_get_data_nonexistent_id(self):
        # Test with nonexistent identifier
        nonexistent_id = TestSourceIdentifier("nonexistent")

        with self.assertRaises(KeyError):
            self.db_manager.import_data(nonexistent_id)


if __name__ == "__main__":
    TestDatabaseManager.run_tests()
