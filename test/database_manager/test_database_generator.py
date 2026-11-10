import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.database_manager.database_errors import DatabaseValidationError
from src.database_manager.database_generator import DatabaseGenerator
from src.database_manager.database_validator import DatabaseValidator
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore


class TestDatabaseGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.db_path = self.root / "index.db"
        SQLiteStore(self.db_path)
        self.generator = DatabaseGenerator()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_index_is_rejected_without_creating_it(self) -> None:
        missing = self.root / "missing" / "index.db"

        with self.assertRaises(FileNotFoundError):
            self.generator.generate_database(missing)

        self.assertFalse(missing.exists())

    def test_validates_by_default(self) -> None:
        with patch.object(DatabaseValidator, "validate_database") as validate:
            db_manager = self.generator.generate_database(self.db_path)

        validate.assert_called_once_with(db_manager)

    def test_validation_can_be_skipped(self) -> None:
        with patch.object(DatabaseValidator, "validate_database") as validate:
            self.generator.generate_database(self.db_path, validate=False)

        validate.assert_not_called()

    def test_validation_failure_propagates(self) -> None:
        MetadataRepository(SQLiteStore(self.db_path)).upsert_record(
            "imu_data", "i1", self.root / "imu_data_i1"
        )

        with self.assertRaises(DatabaseValidationError) as ctx:
            self.generator.generate_database(self.db_path)

        self.assertIn("IMU i1", str(ctx.exception))

    def test_output_dirs_default_to_index_parent(self) -> None:
        db_manager = self.generator.generate_database(self.db_path)

        self.assertEqual(db_manager.io.feature_output_dir, self.root)
        self.assertEqual(db_manager.io.instrument_spec_output_dir, self.root)

    def test_explicit_output_dirs_are_used(self) -> None:
        feature_dir = self.root / "features"

        db_manager = self.generator.generate_database(
            self.db_path, feature_output_dir=feature_dir
        )

        self.assertEqual(db_manager.io.feature_output_dir, feature_dir)
        self.assertEqual(db_manager.io.instrument_spec_output_dir, self.root)


if __name__ == "__main__":
    unittest.main()
