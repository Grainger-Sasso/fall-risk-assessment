import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_errors import (
    DatabaseReadError,
    DatabaseWriteError,
    MetadataIndexError,
)
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


class _FakeRecordFeatures:
    def __init__(self, feature_id: FeatureIdentifier, imu_id: IMUDataIdentifier):
        self.feature_metadata = SimpleNamespace(
            feature_identifier=feature_id,
            imu_data_identifier=imu_id,
        )


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

    def _index(self, id_type: str, identifier: str) -> Path:
        path = self.root / f"{id_type}_{identifier}"
        path.mkdir(parents=True, exist_ok=True)
        self.repository.upsert_record(id_type, identifier, path)
        return path

    def _relate(
        self, source: tuple[str, str], target: tuple[str, str], relation_type: str
    ) -> None:
        """Index both endpoints, then add the relation."""
        for id_type, identifier in (source, target):
            if not self.repository.record_exists(id_type, identifier):
                self._index(id_type, identifier)
        self.repository.add_relation(
            source_type=source[0],
            source_id=source[1],
            target_type=target[0],
            target_id=target[1],
            relation_type=relation_type,
        )

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

    def test_load_instrument_spec_uses_indexed_path(self):
        spec_id = InstrumentSpecificationIdentifier("ltmm_spec_v1")
        spec_path = self._index("instrument_specification", spec_id.value)
        self.io_router.import_instrument_spec.return_value = "spec_loaded"

        result = self.db_manager.load_instrument_spec(spec_id)

        self.io_router.import_instrument_spec.assert_called_once_with(spec_id, spec_path)
        self.assertEqual(result, "spec_loaded")

    def test_get_user_for_imu_resolves_indexed_relation(self):
        imu_id = IMUDataIdentifier("imu_1")
        user_id = UserIdentifier("user_1")
        self._index("imu_data", imu_id.value)
        self._index("user_data", user_id.value)
        self.repository.add_relation(
            source_type="imu_data",
            source_id=imu_id.value,
            target_type="user_data",
            target_id=user_id.value,
            relation_type=IMU_TO_USER,
        )

        self.assertEqual(self.db_manager.get_user_for_imu(imu_id), user_id)
        self.assertIsNone(self.db_manager.get_user_for_imu(IMUDataIdentifier("imu_unlinked")))

    def test_get_instrument_spec_for_imu_resolves_indexed_relation(self):
        imu_id = IMUDataIdentifier("imu_with_spec")
        spec_id = InstrumentSpecificationIdentifier("ltmm_spec_v1")
        self._index("imu_data", imu_id.value)
        self._index("instrument_specification", spec_id.value)
        self.repository.add_relation(
            source_type="imu_data",
            source_id=imu_id.value,
            target_type="instrument_specification",
            target_id=spec_id.value,
            relation_type=IMU_TO_INSTRUMENT_SPEC,
        )

        self.assertEqual(self.db_manager.get_instrument_spec_for_imu(imu_id), spec_id)
        self.assertIsNone(
            self.db_manager.get_instrument_spec_for_imu(IMUDataIdentifier("imu_no_spec"))
        )

    def test_save_features_and_query_features_for_imu(self):
        feature_id = FeatureIdentifier("feature_1")
        imu_id = IMUDataIdentifier("imu_1")
        self._index("imu_data", imu_id.value)
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

    def test_save_features_rolls_back_record_and_payload_on_relation_failure(self):
        feature_id = FeatureIdentifier("feature_rollback")
        imu_id = IMUDataIdentifier("imu_rollback_features")
        self._index("imu_data", imu_id.value)
        exported_path = self.root / "feature_output" / "features_rollback"
        exported_path.mkdir(parents=True, exist_ok=True)
        self.io_router.export_features.return_value = exported_path

        with patch.object(self.repository, "add_relation", side_effect=RuntimeError("boom")):
            with self.assertRaises(DatabaseWriteError):
                self.db_manager.save_features(_FakeRecordFeatures(feature_id, imu_id))

        self.assertFalse(exported_path.exists())
        self.assertFalse(self.repository.record_exists("feature", feature_id.value))

    def test_save_features_rolls_back_when_imu_not_indexed(self):
        feature_id = FeatureIdentifier("feature_orphan")
        imu_id = IMUDataIdentifier("imu_not_indexed")
        exported_path = self.root / "feature_output" / "features_orphan"
        exported_path.mkdir(parents=True, exist_ok=True)
        self.io_router.export_features.return_value = exported_path

        with self.assertRaises(DatabaseWriteError):
            self.db_manager.save_features(_FakeRecordFeatures(feature_id, imu_id))

        self.assertFalse(exported_path.exists())
        self.assertFalse(self.repository.record_exists("feature", feature_id.value))

    def test_delete_features_by_ids_removes_record_relations_and_payload(self):
        feature_id = FeatureIdentifier("feature_cleanup_1")
        imu_id = IMUDataIdentifier("imu_cleanup_1")
        self._index("imu_data", imu_id.value)
        fake_data = _FakeRecordFeatures(feature_id, imu_id)
        exported_path = self.root / "feature_output" / "features_cleanup_1"
        exported_path.mkdir(parents=True, exist_ok=True)
        (exported_path / "features.h5").write_text("dummy")
        self.io_router.export_features.return_value = exported_path
        self.db_manager.save_features(fake_data)

        deleted = self.db_manager.delete_features_by_ids([feature_id], delete_payloads=True)

        self.assertEqual([item.value for item in deleted], [feature_id.value])
        self.assertFalse(
            self.repository.record_exists("feature", feature_id.value)
        )
        self.assertEqual(
            self.repository.get_targets(
                source_type="feature",
                source_id=feature_id.value,
                relation_type=FEATURE_TO_IMU,
            ),
            [],
        )
        self.assertEqual(self.db_manager.get_features_for_imu(imu_id), [])
        self.assertTrue(self.repository.record_exists("imu_data", imu_id.value))
        self.assertFalse(exported_path.exists())

    def test_cleanup_all_features_only_removes_feature_records(self):
        feature_1 = FeatureIdentifier("feature_cleanup_2")
        feature_2 = FeatureIdentifier("feature_cleanup_3")
        imu_id = IMUDataIdentifier("imu_keep")
        user_id = UserIdentifier("user_keep")

        imu_path = self.root / "imu_output" / "imu_keep"
        imu_path.mkdir(parents=True, exist_ok=True)
        self.repository.upsert_record("imu_data", imu_id.value, imu_path)
        user_path = self.root / "user_output" / "user_keep"
        user_path.mkdir(parents=True, exist_ok=True)
        self.repository.upsert_record("user_data", user_id.value, user_path)

        feature_path_1 = self.root / "feature_output" / "feature_cleanup_2"
        feature_path_1.mkdir(parents=True, exist_ok=True)
        self.repository.upsert_record("feature", feature_1.value, feature_path_1)
        self.repository.add_relation(
            source_type="feature",
            source_id=feature_1.value,
            target_type="imu_data",
            target_id=imu_id.value,
            relation_type=FEATURE_TO_IMU,
        )

        feature_path_2 = self.root / "feature_output" / "feature_cleanup_3"
        feature_path_2.mkdir(parents=True, exist_ok=True)
        self.repository.upsert_record("feature", feature_2.value, feature_path_2)
        self.repository.add_relation(
            source_type="feature",
            source_id=feature_2.value,
            target_type="imu_data",
            target_id=imu_id.value,
            relation_type=FEATURE_TO_IMU,
        )

        deleted = self.db_manager.cleanup_all_features(delete_payloads=True)

        self.assertEqual(sorted([item.value for item in deleted]), sorted([feature_1.value, feature_2.value]))
        self.assertTrue(self.repository.record_exists("imu_data", imu_id.value))
        self.assertTrue(self.repository.record_exists("user_data", user_id.value))
        self.assertEqual(self.repository.list_record_ids("feature"), [])
        self.assertEqual(self.repository.get_sources("imu_data", imu_id.value), [])

    def test_load_user_uses_indexed_path(self):
        user_id = UserIdentifier("user_1")
        user_path = self._index("user_data", user_id.value)
        user_data = SimpleNamespace(user_identifier=UserIdentifier("user_1"))
        self.io_router.import_user.return_value = user_data

        result = self.db_manager.load_user(user_id)

        self.io_router.import_user.assert_called_once_with(user_path)
        self.assertIs(result, user_data)

    def test_load_user_rejects_payload_with_different_identifier(self):
        user_id = UserIdentifier("user_expected")
        self._index("user_data", user_id.value)
        self.io_router.import_user.return_value = SimpleNamespace(
            user_identifier=UserIdentifier("user_other")
        )

        with self.assertRaises(DatabaseReadError) as ctx:
            self.db_manager.load_user(user_id)

        self.assertIn("does not match", str(ctx.exception))

    def test_load_features_uses_indexed_path(self):
        feature_id = FeatureIdentifier("feature_1")
        feature_path = self._index("feature", feature_id.value)
        self.io_router.import_features.return_value = "features_loaded"

        result = self.db_manager.load_features(feature_id)

        self.io_router.import_features.assert_called_once_with(feature_id, feature_path)
        self.assertEqual(result, "features_loaded")

    def test_load_methods_raise_read_error_for_unindexed_records(self):
        cases = {
            "load_imu": (self.db_manager.load_imu, IMUDataIdentifier("missing"), "import_imu"),
            "load_user": (self.db_manager.load_user, UserIdentifier("missing"), "import_user"),
            "load_features": (
                self.db_manager.load_features,
                FeatureIdentifier("missing"),
                "import_features",
            ),
            "load_instrument_spec": (
                self.db_manager.load_instrument_spec,
                InstrumentSpecificationIdentifier("missing"),
                "import_instrument_spec",
            ),
        }
        for name, (load, identifier, importer) in cases.items():
            with self.subTest(method=name):
                with self.assertRaises(DatabaseReadError) as ctx:
                    load(identifier)
                self.assertIsInstance(ctx.exception.__cause__, MetadataIndexError)
                getattr(self.io_router, importer).assert_not_called()

    def test_load_methods_wrap_importer_failures_in_read_error(self):
        cases = {
            "load_imu": (
                self.db_manager.load_imu, IMUDataIdentifier("i1"), "imu_data", "import_imu"
            ),
            "load_user": (
                self.db_manager.load_user, UserIdentifier("u1"), "user_data", "import_user"
            ),
            "load_features": (
                self.db_manager.load_features,
                FeatureIdentifier("f1"),
                "feature",
                "import_features",
            ),
            "load_instrument_spec": (
                self.db_manager.load_instrument_spec,
                InstrumentSpecificationIdentifier("s1"),
                "instrument_specification",
                "import_instrument_spec",
            ),
        }
        for name, (load, identifier, id_type, importer) in cases.items():
            with self.subTest(method=name):
                self._index(id_type, identifier.value)
                getattr(self.io_router, importer).side_effect = OSError("unreadable payload")
                with self.assertRaises(DatabaseReadError) as ctx:
                    load(identifier)
                self.assertIn("unreadable payload", str(ctx.exception))

    def test_list_ids_return_sorted_typed_identifiers_per_type(self):
        for identifier in ["imu_b", "imu_a"]:
            self._index("imu_data", identifier)
        self._index("user_data", "user_1")
        self._index("instrument_specification", "spec_1")
        self._index("feature", "feature_1")
        cases = {
            "list_imu_ids": (IMUDataIdentifier, ["imu_a", "imu_b"]),
            "list_user_ids": (UserIdentifier, ["user_1"]),
            "list_instrument_spec_ids": (InstrumentSpecificationIdentifier, ["spec_1"]),
            "list_feature_ids": (FeatureIdentifier, ["feature_1"]),
        }
        for method, (id_class, expected) in cases.items():
            with self.subTest(method=method):
                ids = getattr(self.db_manager, method)()
                self.assertEqual([item.value for item in ids], expected)
                for item in ids:
                    self.assertIs(type(item), id_class)

    def test_list_ids_are_empty_for_empty_index(self):
        for method in [
            "list_imu_ids",
            "list_user_ids",
            "list_instrument_spec_ids",
            "list_feature_ids",
        ]:
            with self.subTest(method=method):
                self.assertEqual(getattr(self.db_manager, method)(), [])

    def test_get_imu_for_feature_resolves_indexed_relation(self):
        self._relate(("feature", "f1"), ("imu_data", "i1"), FEATURE_TO_IMU)

        self.assertEqual(
            self.db_manager.get_imu_for_feature(FeatureIdentifier("f1")),
            IMUDataIdentifier("i1"),
        )
        self.assertIsNone(self.db_manager.get_imu_for_feature(FeatureIdentifier("f_unlinked")))

    def test_get_features_for_imu_ignores_non_feature_sources(self):
        self._relate(("feature", "f1"), ("imu_data", "i1"), FEATURE_TO_IMU)
        self._relate(("user_data", "u1"), ("imu_data", "i1"), FEATURE_TO_IMU)

        features = self.db_manager.get_features_for_imu(IMUDataIdentifier("i1"))

        self.assertEqual(features, [FeatureIdentifier("f1")])

    def test_delete_features_by_ids_skips_unknown_ids(self):
        known = FeatureIdentifier("feature_known")
        unknown = FeatureIdentifier("feature_unknown")
        self._relate(("feature", known.value), ("imu_data", "i1"), FEATURE_TO_IMU)

        deleted = self.db_manager.delete_features_by_ids([unknown, known])

        self.assertEqual(deleted, [known])
        self.assertFalse(self.repository.record_exists("feature", known.value))
        self.assertEqual(self.db_manager.delete_features_by_ids([unknown]), [])

    def test_delete_features_without_payloads_keeps_files_on_disk(self):
        feature_id = FeatureIdentifier("feature_keep_payload")
        self._relate(("feature", feature_id.value), ("imu_data", "i1"), FEATURE_TO_IMU)
        payload_path = self.repository.get_record_path("feature", feature_id.value)
        (payload_path / "features.h5").write_text("dummy")

        deleted = self.db_manager.delete_features_by_ids([feature_id], delete_payloads=False)

        self.assertEqual(deleted, [feature_id])
        self.assertFalse(self.repository.record_exists("feature", feature_id.value))
        self.assertEqual(self.db_manager.get_features_for_imu(IMUDataIdentifier("i1")), [])
        self.assertTrue((payload_path / "features.h5").exists())

    def test_cleanup_all_features_without_payloads_keeps_files_on_disk(self):
        self._relate(("feature", "f1"), ("imu_data", "i1"), FEATURE_TO_IMU)
        payload_path = self.repository.get_record_path("feature", "f1")

        deleted = self.db_manager.cleanup_all_features(delete_payloads=False)

        self.assertEqual(deleted, [FeatureIdentifier("f1")])
        self.assertEqual(self.db_manager.list_feature_ids(), [])
        self.assertTrue(payload_path.exists())

    def test_lookups_reject_relations_pointing_at_wrong_record_type(self):
        cases = {
            "get_user_for_imu": (
                ("imu_data", "i1"),
                ("feature", "wrong"),
                IMU_TO_USER,
                lambda: self.db_manager.get_user_for_imu(IMUDataIdentifier("i1")),
            ),
            "get_imu_for_feature": (
                ("feature", "f1"),
                ("user_data", "wrong"),
                FEATURE_TO_IMU,
                lambda: self.db_manager.get_imu_for_feature(FeatureIdentifier("f1")),
            ),
            "get_instrument_spec_for_imu": (
                ("imu_data", "i2"),
                ("user_data", "wrong"),
                IMU_TO_INSTRUMENT_SPEC,
                lambda: self.db_manager.get_instrument_spec_for_imu(IMUDataIdentifier("i2")),
            ),
        }
        for name, (source, target, relation_type, lookup) in cases.items():
            with self.subTest(method=name):
                self._relate(source, target, relation_type)
                with self.assertRaises(MetadataIndexError) as ctx:
                    lookup()
                self.assertIn("Expected", str(ctx.exception))

    def test_lookup_rejects_relation_to_unknown_record_type(self):
        self._relate(("imu_data", "i1"), ("mystery_type", "m1"), IMU_TO_USER)

        with self.assertRaises(MetadataIndexError) as ctx:
            self.db_manager.get_user_for_imu(IMUDataIdentifier("i1"))

        self.assertIn("mystery_type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
