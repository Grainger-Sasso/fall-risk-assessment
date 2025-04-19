import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Type
from unittest.mock import MagicMock, call

from src.data_io.import_export.exporters.database_exporter import DatabaseExporter
from src.data_io.import_export.exporters.exporter import Exporter
from src.data_io.import_export.exporters.mapping.mapping_exporter import MappingExporter
from src.data_io.import_export.exporters.registry.registry_exporter import (
    RegistryExporter,
)
from src.data_io.import_export.importers.importer import Importer
from src.data_io.import_export.importers.mapping.mapping_importer import MappingImporter
from src.data_io.import_export.importers.registry.registry_importer import (
    RegistryImporter,
)
from src.database_manager.data_access.export_manager import ExportManager
from src.database_manager.data_access.import_manager import ImportManager
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.output_directory_manager import (
    OutputDirectoryManager,
)
from src.database_manager.data_access.registry_manager import RegistryManager
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FeatureDataHelper
from test.data_io.test_data.test_data_helper import TestConstants as DataIOTestConstants
from test.database_manager.test_data.test_data_helper import (
    DatabaseManagerTestHelper,
    TestConstants,
    TestSourceIdentifier,
    TestTargetIdentifier,
)


class TestDatabaseManager(BaseTest):
    def setUp(self):
        self.helper = DatabaseManagerTestHelper()
        self.feature_data_helper = FeatureDataHelper()

        self.registry = self.helper.create_test_registry()

        # Create temp directories for test
        self.raw_feature_temp_dir = tempfile.mkdtemp()
        self.agg_feature_temp_dir = tempfile.mkdtemp()
        self.raw_feature_temp_path = Path(self.raw_feature_temp_dir)
        self.agg_feature_temp_path = Path(self.agg_feature_temp_dir)

        # Create empty registry and mapping for raw features
        self.raw_feature_registry = Registry(
            {},
            id_type=RawFeatureIdentifier,
            subdir_path=self.raw_feature_temp_path,
        )
        self.raw_feature_mapping = Mapping(
            {},
            source_id_type=RawFeatureIdentifier,
            target_id_type=IMUDataIdentifier,
            subdir_path=self.raw_feature_temp_path,
        )

        # Create empty registry and mapping for aggregate features
        self.agg_feature_registry = Registry(
            {},
            id_type=AggregateFeatureIdentifier,
            subdir_path=self.agg_feature_temp_path,
        )
        self.agg_feature_mapping = Mapping(
            {},
            source_id_type=AggregateFeatureIdentifier,
            target_id_type=RawFeatureIdentifier,
            subdir_path=self.agg_feature_temp_path,
        )

        # Create registry and mapping managers
        self.registry_manager = RegistryManager(
            {
                TestSourceIdentifier: self.registry,
                RawFeatureIdentifier: self.raw_feature_registry,
                AggregateFeatureIdentifier: self.agg_feature_registry,
            }
        )
        self.mapping = self.helper.create_test_mapping()
        self.mapping_manager = MappingManager(
            {
                TestSourceIdentifier: self.mapping,
                RawFeatureIdentifier: self.raw_feature_mapping,
                AggregateFeatureIdentifier: self.agg_feature_mapping,
            }
        )

        # Export registry and mapping to temp directories
        self.registry_exporter = RegistryExporter()
        self.mapping_exporter = MappingExporter()

        # Export raw feature files
        self.registry_exporter.export_data(
            self.raw_feature_temp_path, self.raw_feature_registry
        )
        self.mapping_exporter.export_data(
            self.raw_feature_temp_path, self.raw_feature_mapping
        )

        # Export aggregate feature files
        self.registry_exporter.export_data(
            self.agg_feature_temp_path, self.agg_feature_registry
        )
        self.mapping_exporter.export_data(
            self.agg_feature_temp_path, self.agg_feature_mapping
        )

        # Create mock importer
        self.mock_importer = MagicMock(spec=Importer)
        self.mock_importer.import_data.return_value = "test_data"

        # Create import manager with mock importer
        self.import_manager = ImportManager({TestSourceIdentifier: self.mock_importer})

        # Create mock exporter and database exporter
        self.test_registry_path = Path("test_registry_path")
        self.mock_exporter = MagicMock(spec=Exporter)
        self.mock_exporter.export_data.return_value = self.test_registry_path

        # Create mock export manager
        self.export_manager = ExportManager(
            {
                RawFeatureIdentifier: self.mock_exporter,
                AggregateFeatureIdentifier: self.mock_exporter,
            }
        )

        # Create mock output dir manager
        self.mock_output_dir_manager = MagicMock(spec=OutputDirectoryManager)
        self.mock_output_dir_manager.get_provider.return_value = (
            TestConstants.TEST_PATHS.value[0]
        )

        # Create database manager
        self.db_manager = DatabaseManager(
            registry_manager=self.registry_manager,
            mapping_manager=self.mapping_manager,
            import_manager=self.import_manager,
            export_manager=self.export_manager,
            output_dir_manager=self.mock_output_dir_manager,
        )
        self.registry_importer = RegistryImporter()
        self.mapping_importer = MappingImporter()

    def tearDown(self):
        # Clean up raw feature test directory
        if self.raw_feature_temp_path.exists():
            for root, dirs, files in os.walk(self.raw_feature_temp_path, topdown=False):
                for name in files:
                    (Path(root) / name).unlink()
                for name in dirs:
                    (Path(root) / name).rmdir()
            self.raw_feature_temp_path.rmdir()

        # Clean up aggregate feature test directory
        if self.agg_feature_temp_path.exists():
            for root, dirs, files in os.walk(self.agg_feature_temp_path, topdown=False):
                for name in files:
                    (Path(root) / name).unlink()
                for name in dirs:
                    (Path(root) / name).rmdir()
            self.agg_feature_temp_path.rmdir()

    def test_import_data(self):
        # Create test identifier
        test_id = self.helper.create_test_identifier()

        # Get data using database manager
        data = self.db_manager.import_data([test_id])

        # Verify correct path was retrieved from registry
        expected_path = self.registry.get_path_from_id(test_id)

        # Verify importer was called with correct path
        self.mock_importer.import_data.assert_called_once_with(expected_path)

        # Verify returned data matches mock importer output
        self.assertEqual(data, ["test_data"])

    def test_import_data_invalid_id_type(self):
        # Test with invalid identifier type
        invalid_id = TestTargetIdentifier("invalid")

        with self.assertRaises(KeyError):
            self.db_manager.import_data([invalid_id])

    def test_import_data_nonexistent_id(self):
        # Test with nonexistent identifier
        nonexistent_id = TestSourceIdentifier("nonexistent")

        with self.assertRaises(KeyError):
            self.db_manager.import_data([nonexistent_id])

    def test_export_data(self):
        # Assert the registry/mapping manager entries are none
        imported_raw_feature_registry: Registry = self.registry_importer.import_data(
            self.raw_feature_temp_path, RawFeatureIdentifier
        )
        self.assertTrue(not imported_raw_feature_registry.registry)
        imported_raw_feature_mapping: Mapping = self.mapping_importer.import_data(
            self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
        )
        self.assertTrue(not imported_raw_feature_mapping.map)
        # Import registry/mappings and assert none
        self.assertTrue(
            not self.registry_manager.get_provider(RawFeatureIdentifier).registry
        )
        self.assertTrue(not self.mapping_manager.get_provider(RawFeatureIdentifier).map)
        # Create test data to export
        test_raw_feature_1 = self.feature_data_helper.create_test_raw_feature()
        test_raw_feature_2 = self.feature_data_helper.create_test_raw_feature()
        test_raw_feature_2.metadata.raw_feature_identifier = RawFeatureIdentifier(
            DataIOTestConstants.RAW_FEATURE_ID_2.value
        )
        raw_feature_list = [test_raw_feature_1, test_raw_feature_2]
        # Export
        self.db_manager.export_data(raw_feature_list)
        # Verify the correct exporter is called with expected path and data
        expected_parent_dir = TestConstants.TEST_PATHS.value[0]
        self.mock_exporter.export_data.assert_has_calls([
            call(expected_parent_dir, raw_feature_list[0]),
            call(expected_parent_dir, raw_feature_list[1])
        ])
        # Assert the registry/mappings properly updated
        imported_raw_feature_registry: Registry = self.registry_importer.import_data(
            self.raw_feature_temp_path, RawFeatureIdentifier
        )
        self.assertEqual(
            imported_raw_feature_registry.registry,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: self.test_registry_path,
            },
        )
        imported_raw_feature_mapping: Mapping = self.mapping_importer.import_data(
            self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
        )
        self.assertEqual(
            imported_raw_feature_mapping.map,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value
            },
        )
        # Assert the registry/mappings managers properly updated
        manager_raw_feature_registry: Registry = self.registry_manager.get_provider(
            RawFeatureIdentifier
        )
        self.assertEqual(
            manager_raw_feature_registry.registry,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: self.test_registry_path,
            },
        )
        manager_raw_feature_mapping: Mapping = self.mapping_manager.get_provider(
            RawFeatureIdentifier
        )
        self.assertEqual(
            manager_raw_feature_mapping.map,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value
            },
        )
        # Test exporting additional list
        test_raw_feature_3 = self.feature_data_helper.create_test_raw_feature()
        test_raw_feature_4 = self.feature_data_helper.create_test_raw_feature()
        test_raw_feature_3.metadata.raw_feature_identifier = RawFeatureIdentifier(
            DataIOTestConstants.RAW_FEATURE_ID_3.value
        )
        test_raw_feature_4.metadata.raw_feature_identifier = RawFeatureIdentifier(
            DataIOTestConstants.RAW_FEATURE_ID_4.value
        )
        raw_feature_list_2 = [test_raw_feature_3, test_raw_feature_4]
        self.db_manager.export_data(raw_feature_list_2)
        # Verify the correct exporter is called with expected path and data
        expected_parent_dir = TestConstants.TEST_PATHS.value[0]
        self.mock_exporter.export_data.assert_has_calls([
            call(expected_parent_dir, raw_feature_list_2[0]),
            call(expected_parent_dir, raw_feature_list_2[1])
        ])
        # Assert the registry/mappings properly updated
        imported_raw_feature_registry: Registry = self.registry_importer.import_data(
            self.raw_feature_temp_path, RawFeatureIdentifier
        )
        self.assertEqual(
            imported_raw_feature_registry.registry,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_3.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_4.value: self.test_registry_path,
            },
        )
        imported_raw_feature_mapping: Mapping = self.mapping_importer.import_data(
            self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
        )
        self.assertEqual(
            imported_raw_feature_mapping.map,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_3.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_4.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
            },
        )
        # Assert the registry/mappings managers properly updated
        manager_raw_feature_registry: Registry = self.registry_manager.get_provider(
            RawFeatureIdentifier
        )
        self.assertEqual(
            manager_raw_feature_registry.registry,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_3.value: self.test_registry_path,
                DataIOTestConstants.RAW_FEATURE_ID_4.value: self.test_registry_path,
            },
        )
        manager_raw_feature_mapping: Mapping = self.mapping_manager.get_provider(
            RawFeatureIdentifier
        )
        self.assertEqual(
            manager_raw_feature_mapping.map,
            {
                DataIOTestConstants.RAW_FEATURE_ID.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_2.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_3.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
                DataIOTestConstants.RAW_FEATURE_ID_4.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
            },
        )


    # def test_export_raw_feature_list(self):
    #     # Create test raw feature list
    #     raw_feature_list = [self.feature_data_helper.create_test_raw_feature()]
    #     # Assert the registry/mapping manager entries are none
    #     imported_raw_feature_registry: Registry = self.registry_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_registry.registry)
    #     imported_raw_feature_mapping: Mapping = self.mapping_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_mapping.map)
    #     # Import registry/mappings and assert none
    #     self.assertTrue(
    #         not self.registry_manager.get_provider(RawFeatureIdentifier).registry
    #     )
    #     self.assertTrue(not self.mapping_manager.get_provider(RawFeatureIdentifier).map)
    #     # Call method under test
    #     self.db_manager.export_raw_feature_list(raw_feature_list)
    #     # Verify the correct exporter is called with expected path and data
    #     expected_parent_dir = TestConstants.TEST_PATHS.value[0]
    #     self.mock_exporter.export_data.assert_called_once_with(
    #         expected_parent_dir, raw_feature_list[0]
    #     )
    #     # Assert the registry/mappings properly updated
    #     imported_raw_feature_registry: Registry = self.registry_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier
    #     )
    #     self.assertEqual(
    #         imported_raw_feature_registry.registry,
    #         {DataIOTestConstants.RAW_FEATURE_ID.value: self.test_registry_path},
    #     )
    #     imported_raw_feature_mapping: Mapping = self.mapping_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertEqual(
    #         imported_raw_feature_mapping.map,
    #         {
    #             DataIOTestConstants.RAW_FEATURE_ID.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value
    #         },
    #     )
    #     # Assert the registry/mappings managers properly updated
    #     manager_raw_feature_registry: Registry = self.registry_manager.get_provider(
    #         RawFeatureIdentifier
    #     )
    #     self.assertEqual(
    #         manager_raw_feature_registry.registry,
    #         {DataIOTestConstants.RAW_FEATURE_ID.value: self.test_registry_path},
    #     )
    #     manager_raw_feature_mapping: Mapping = self.mapping_manager.get_provider(
    #         RawFeatureIdentifier
    #     )
    #     self.assertEqual(
    #         manager_raw_feature_mapping.map,
    #         {
    #             DataIOTestConstants.RAW_FEATURE_ID.value: DataIOTestConstants.FEATURE_IMU_DATA_ID.value,
    #         },
    #     )

    # def test_export_agg_feature_list(self):
    #     # Create test agg feature list
    #     agg_feature_list = [self.feature_data_helper.create_test_aggregate_feature()]
    #     # Assert the registry/mapping manager entries are none
    #     imported_agg_feature_registry: Registry = self.registry_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_registry.registry)
    #     imported_agg_feature_mapping: Mapping = self.mapping_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_mapping.map)
    #     # Import registry/mappings and assert none
    #     self.assertTrue(
    #         not self.registry_manager.get_provider(AggregateFeatureIdentifier).registry
    #     )
    #     self.assertTrue(
    #         not self.mapping_manager.get_provider(AggregateFeatureIdentifier).map
    #     )
    #     # Call method under test
    #     self.db_manager.export_aggregate_feature_list(agg_feature_list)
    #     # Verify the correct exporter is called with expected path and data
    #     expected_parent_dir = TestConstants.TEST_PATHS.value[0]
    #     self.mock_exporter.export_data.assert_called_once_with(
    #         expected_parent_dir, agg_feature_list[0]
    #     )
    #     # Assert the registry/mappings properly updated
    #     imported_agg_feature_registry: Registry = self.registry_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier
    #     )
    #     self.assertEqual(
    #         imported_agg_feature_registry.registry,
    #         {DataIOTestConstants.AGG_FEATURE_ID.value: self.test_registry_path},
    #     )
    #     imported_agg_feature_mapping: Mapping = self.mapping_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertEqual(
    #         imported_agg_feature_mapping.map,
    #         {
    #             DataIOTestConstants.AGG_FEATURE_ID.value: DataIOTestConstants.RAW_FEATURE_ID.value
    #         },
    #     )
    #     # Assert the registry/mappings managers properly updated
    #     manager_agg_feature_registry: Registry = self.registry_manager.get_provider(
    #         AggregateFeatureIdentifier
    #     )
    #     self.assertEqual(
    #         manager_agg_feature_registry.registry,
    #         {DataIOTestConstants.AGG_FEATURE_ID.value: self.test_registry_path},
    #     )
    #     manager_agg_feature_mapping: Mapping = self.mapping_manager.get_provider(
    #         AggregateFeatureIdentifier
    #     )
    #     self.assertEqual(
    #         manager_agg_feature_mapping.map,
    #         {
    #             DataIOTestConstants.AGG_FEATURE_ID.value: DataIOTestConstants.RAW_FEATURE_ID.value,
    #         },
    #     )

    # def test_export_raw_feature_list_export_fails(self):
    #     # Create test raw feature list
    #     raw_feature_list = [self.feature_data_helper.create_test_raw_feature()]

    #     # Configure mock exporter to raise exception
    #     self.mock_exporter.export_data.side_effect = Exception("Export failed")

    #     # Call method and verify exception
    #     with self.assertRaises(Exception) as context:
    #         self.db_manager.export_raw_feature_list(raw_feature_list)

    #     self.assertIn("Export failed", str(context.exception))

    #     # Verify registry and mapping were not updated
    #     imported_raw_feature_registry = self.registry_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_registry.registry)
    #     imported_raw_feature_mapping = self.mapping_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_mapping.map)

    # def test_export_raw_feature_list_empty_list(self):
    #     # Call method with empty list
    #     with self.assertRaises(ValueError) as context:
    #         self.db_manager.export_raw_feature_list([])

    #     self.assertIn("Feature list cannot be empty", str(context.exception))

    #     # Verify registry and mapping were not updated
    #     imported_raw_feature_registry = self.registry_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_registry.registry)
    #     imported_raw_feature_mapping = self.mapping_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_mapping.map)

    # def test_export_agg_feature_list_export_fails(self):
    #     # Create test aggregate feature list
    #     agg_feature_list = [self.feature_data_helper.create_test_aggregate_feature()]

    #     # Configure mock exporter to raise exception
    #     self.mock_exporter.export_data.side_effect = Exception("Export failed")

    #     # Call method and verify exception
    #     with self.assertRaises(Exception) as context:
    #         self.db_manager.export_aggregate_feature_list(agg_feature_list)

    #     self.assertIn("Export failed", str(context.exception))

    #     # Verify registry and mapping were not updated
    #     imported_agg_feature_registry = self.registry_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_registry.registry)
    #     imported_agg_feature_mapping = self.mapping_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_mapping.map)

    # def test_export_agg_feature_list_empty_list(self):
    #     # Call method with empty list
    #     with self.assertRaises(ValueError) as context:
    #         self.db_manager.export_aggregate_feature_list([])

    #     self.assertIn("Feature list cannot be empty", str(context.exception))

    #     # Verify registry and mapping were not updated
    #     imported_agg_feature_registry = self.registry_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_registry.registry)
    #     imported_agg_feature_mapping = self.mapping_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_mapping.map)

    # def test_export_raw_feature_list_invalid_output_dir(self):
    #     # Create test raw feature list
    #     raw_feature_list = [self.feature_data_helper.create_test_raw_feature()]

    #     # Configure output dir manager to return None
    #     self.mock_output_dir_manager.get_provider.return_value = None

    #     # Call method and verify exception
    #     with self.assertRaises(ValueError) as context:
    #         self.db_manager.export_raw_feature_list(raw_feature_list)

    #     self.assertIn("No output directory configured", str(context.exception))

    #     # Verify registry and mapping were not updated
    #     imported_raw_feature_registry = self.registry_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_registry.registry)
    #     imported_raw_feature_mapping = self.mapping_importer.import_data(
    #         self.raw_feature_temp_path, RawFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_raw_feature_mapping.map)

    # def test_export_agg_feature_list_invalid_output_dir(self):
    #     # Create test aggregate feature list
    #     agg_feature_list = [self.feature_data_helper.create_test_aggregate_feature()]

    #     # Configure output dir manager to return None
    #     self.mock_output_dir_manager.get_provider.return_value = None

    #     # Call method and verify exception
    #     with self.assertRaises(ValueError) as context:
    #         self.db_manager.export_aggregate_feature_list(agg_feature_list)

    #     self.assertIn("No output directory configured", str(context.exception))

    #     # Verify registry and mapping were not updated
    #     imported_agg_feature_registry = self.registry_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_registry.registry)
    #     imported_agg_feature_mapping = self.mapping_importer.import_data(
    #         self.agg_feature_temp_path, AggregateFeatureIdentifier, IMUDataIdentifier
    #     )
    #     self.assertTrue(not imported_agg_feature_mapping.map)


if __name__ == "__main__":
    TestDatabaseManager.run_tests()
