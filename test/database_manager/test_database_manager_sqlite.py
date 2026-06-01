import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_errors import DatabaseWriteError
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.relation_types import (
    FEATURE_TO_IMU,
    IMU_TO_INSTRUMENT_SPEC,
    IMU_TO_USER,
)
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier


class _FakeIMUData:
    def __init__(self, source_id: IMUDataIdentifier, user_id: UserIdentifier):
        self.metadata = SimpleNamespace(
            imu_data_identifier=source_id,
            user_identifier=user_id,
        )


class _FakeRecordFeatures:
    def __init__(self, feature_id: FeatureIdentifier, imu_id: IMUDataIdentifier):
        self.feature_metadata = SimpleNamespace(
            feature_identifier=feature_id,
            imu_data_identifier=imu_id,
        )


class _FakeInstrumentSpec:
    def __init__(self, spec_id: InstrumentSpecificationIdentifier):
        self.specification_id = spec_id


class TestDatabaseManagerSQLite(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        repository = MetadataRepository(SQLiteStore(self.root / "index.db"))
        self.repository = repository
        self.io_router = MagicMock(spec=DomainIORouter)
        self.db_manager = DatabaseManager(
            metadata_repository=self.repository,
            io_router=self.io_router,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_imu_uses_sqlite_registry_path(self):
        imu_id = IMUDataIdentifier("imu_1")
        imu_path = self.root / "imu_data_imu_1"
        imu_path.mkdir(parents=True, exist_ok=True)
        (imu_path / "imu_data.h5").write_text("dummy")

        self.repository.upsert_record("imu_data", imu_id.value, imu_path)
        self.io_router.import_imu.return_value = "imported"

        result = self.db_manager.load_imu(imu_id)

        self.io_router.import_imu.assert_called_once_with(imu_id, imu_path)
        self.assertEqual(result, "imported")

    def test_save_imu_updates_sqlite_record_and_relation(self):
        imu_id = IMUDataIdentifier("imu_2")
        user_id = UserIdentifier("user_2")
        fake_data = _FakeIMUData(imu_id, user_id)

        exported_path = self.root / "imu_output" / "imu_data_imu_2"
        exported_path.mkdir(parents=True, exist_ok=True)
        (exported_path / "imu_data.h5").write_text("dummy")
        self.io_router.export_imu.return_value = exported_path

        self.db_manager.save_imu(fake_data)

        self.assertEqual(self.repository.get_record_path("imu_data", imu_id.value), exported_path)
        targets = self.repository.get_targets(
            source_type="imu_data",
            source_id=imu_id.value,
            relation_type=IMU_TO_USER,
        )
        self.assertEqual(targets, [("user_data", user_id.value)])

        mapped_target = self.db_manager.get_user_for_imu(imu_id)
        self.assertIsNotNone(mapped_target)
        self.assertEqual(mapped_target.value, user_id.value)

    def test_save_features_and_query_features_for_imu(self):
        feature_id = FeatureIdentifier("feature_1")
        imu_id = IMUDataIdentifier("imu_1")
        fake_data = _FakeRecordFeatures(feature_id, imu_id)
        exported_path = self.root / "feature_output" / "features_feature_1"
        exported_path.mkdir(parents=True, exist_ok=True)
        (exported_path / "features.h5").write_text("dummy")
        self.io_router.export_features.return_value = exported_path

        self.db_manager.save_features(fake_data)

        self.assertEqual(
            self.repository.get_record_path("feature", feature_id.value), exported_path
        )
        self.assertTrue(
            self.repository.relation_exists(
                source_type="feature",
                source_id=feature_id.value,
                target_type="imu_data",
                target_id=imu_id.value,
                relation_type=FEATURE_TO_IMU,
            )
        )
        features = self.db_manager.get_features_for_imu(imu_id)
        self.assertEqual([item.value for item in features], [feature_id.value])

    def test_save_imu_rolls_back_payload_on_metadata_failure(self):
        imu_id = IMUDataIdentifier("imu_rollback")
        user_id = UserIdentifier("user_rollback")
        fake_data = _FakeIMUData(imu_id, user_id)
        exported_path = self.root / "imu_output" / "imu_data_rollback"
        exported_path.mkdir(parents=True, exist_ok=True)
        (exported_path / "imu_data.h5").write_text("dummy")
        self.io_router.export_imu.return_value = exported_path

        with patch.object(
            self.repository, "add_relation", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(DatabaseWriteError):
                self.db_manager.save_imu(fake_data)

        self.assertFalse(exported_path.exists())

    def test_save_and_load_instrument_spec(self):
        spec_id = InstrumentSpecificationIdentifier("ltmm_spec_v1")
        spec_data = _FakeInstrumentSpec(spec_id)
        exported_path = self.root / "instrument_spec_output" / "instrument_specification_1"
        exported_path.mkdir(parents=True, exist_ok=True)
        (exported_path / "instrument_spec.json").write_text("dummy")
        self.io_router.export_instrument_spec.return_value = exported_path
        self.io_router.import_instrument_spec.return_value = "spec_loaded"

        self.db_manager.save_instrument_spec(spec_data)

        self.assertEqual(
            self.repository.get_record_path("instrument_specification", spec_id.value),
            exported_path,
        )
        result = self.db_manager.load_instrument_spec(spec_id)
        self.io_router.import_instrument_spec.assert_called_once_with(spec_id, exported_path)
        self.assertEqual(result, "spec_loaded")

    def test_link_and_get_instrument_spec_for_imu(self):
        imu_id = IMUDataIdentifier("imu_with_spec")
        spec_id = InstrumentSpecificationIdentifier("ltmm_spec_v1")

        self.db_manager.link_imu_to_instrument_spec(imu_id, spec_id)

        target = self.db_manager.get_instrument_spec_for_imu(imu_id)
        self.assertIsNotNone(target)
        self.assertEqual(target.value, spec_id.value)
        self.assertTrue(
            self.repository.relation_exists(
                source_type="imu_data",
                source_id=imu_id.value,
                target_type="instrument_specification",
                target_id=spec_id.value,
                relation_type=IMU_TO_INSTRUMENT_SPEC,
            )
        )


if __name__ == "__main__":
    unittest.main()
