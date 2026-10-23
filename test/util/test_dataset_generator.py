import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.database_manager.database_errors import MetadataIndexError
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
from src.util.dataset_generator import dataset_generator
from src.util.dataset_generator.dataset_generator import (
    CLINICAL_DEMO_FILE,
    FEATURE_DATA_FILE,
    IMU_DATA_FILE,
    USER_DATA_FILE,
    build_sql_index,
)

INSTRUMENT_NAME = "lower_back_sensor"


class _FakeIMUImporter:
    """Reads ``<imu_dir>/user.txt`` for the linked user ID; the dir name is the IMU ID."""

    def import_data(self, imu_dir: Path) -> SimpleNamespace:
        user_value = (imu_dir / "user.txt").read_text()
        return SimpleNamespace(
            get_data_id=lambda: IMUDataIdentifier(imu_dir.name),
            get_associated_data_id=lambda: UserIdentifier(user_value),
            metadata=SimpleNamespace(
                instrument_identifier=SimpleNamespace(name=INSTRUMENT_NAME)
            ),
        )


class _FakeUserImporter:
    def import_data(self, user_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(get_data_id=lambda: UserIdentifier(user_dir.name))


class _FakeFeatureImporter:
    def import_data(self, feature_dir: Path) -> SimpleNamespace:
        imu_value = (feature_dir / "imu.txt").read_text()
        return SimpleNamespace(
            get_data_id=lambda: FeatureIdentifier(feature_dir.name),
            get_associated_data_id=lambda: IMUDataIdentifier(imu_value),
        )


class _FakeJSONReader:
    def read(self, path: Path) -> dict:
        return {"spec_id": path.stem}


class _FakeSpecBuilder:
    def build(self, spec_json: dict) -> SimpleNamespace:
        return SimpleNamespace(
            specification_id=InstrumentSpecificationIdentifier(spec_json["spec_id"]),
            imu_name=INSTRUMENT_NAME,
        )


class TestBuildSqlIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.imu_parent = self.root / "imu"
        self.user_parent = self.root / "users"
        self.feature_parent = self.root / "features"
        self.spec_dir = self.root / "specs"
        for directory in (self.imu_parent, self.user_parent, self.feature_parent, self.spec_dir):
            directory.mkdir()
        self.spec_file = self.spec_dir / "spec_1.json"
        self.spec_file.write_text("{}")
        self.db_path = self.root / "index.db"

        self._patches = ExitStack()
        for name, fake in {
            "IMUDataImporter": _FakeIMUImporter,
            "UserDataImporter": _FakeUserImporter,
            "FeatureImporter": _FakeFeatureImporter,
            "JSONDictFileReader": _FakeJSONReader,
            "IMUSpecificationBuilder": _FakeSpecBuilder,
        }.items():
            self._patches.enter_context(patch.object(dataset_generator, name, fake))

    def tearDown(self) -> None:
        self._patches.close()
        self.temp_dir.cleanup()

    def _add_user(self, user_id: str) -> Path:
        user_dir = self.user_parent / user_id
        user_dir.mkdir()
        (user_dir / USER_DATA_FILE).write_text("{}")
        (user_dir / CLINICAL_DEMO_FILE).write_text("{}")
        return user_dir

    def _add_imu(self, imu_id: str, user_id: str) -> Path:
        imu_dir = self.imu_parent / imu_id
        imu_dir.mkdir()
        (imu_dir / IMU_DATA_FILE).write_text("")
        (imu_dir / "user.txt").write_text(user_id)
        return imu_dir

    def _add_feature(self, feature_id: str, imu_id: str) -> Path:
        feature_dir = self.feature_parent / feature_id
        feature_dir.mkdir()
        (feature_dir / FEATURE_DATA_FILE).write_text("")
        (feature_dir / "imu.txt").write_text(imu_id)
        return feature_dir

    def _build(self) -> MetadataRepository:
        has_features = any(self.feature_parent.iterdir())
        build_sql_index(
            imu_parent_dir=self.imu_parent,
            user_parent_dir=self.user_parent,
            instrument_spec_inputs=self.spec_file,
            sqlite_db_path=self.db_path,
            feature_parent_dir=self.feature_parent if has_features else None,
        )
        return MetadataRepository(SQLiteStore(self.db_path))

    def _assert_index_empty(self) -> None:
        repo = MetadataRepository(SQLiteStore(self.db_path))
        for id_type in ("imu_data", "user_data", "feature", "instrument_specification"):
            self.assertEqual(repo.list_record_ids(id_type), [], id_type)

    def test_builds_records_and_relations_for_consistent_dataset(self) -> None:
        user_dir = self._add_user("user_1")
        imu_dir = self._add_imu("imu_1", "user_1")
        feature_dir = self._add_feature("feature_1", "imu_1")

        repo = self._build()

        self.assertEqual(repo.get_record_path("user_data", "user_1"), user_dir.resolve())
        self.assertEqual(repo.get_record_path("imu_data", "imu_1"), imu_dir.resolve())
        self.assertEqual(repo.get_record_path("feature", "feature_1"), feature_dir.resolve())
        self.assertEqual(repo.list_record_ids("instrument_specification"), ["spec_1"])
        self.assertEqual(
            repo.get_targets("imu_data", "imu_1", IMU_TO_USER), [("user_data", "user_1")]
        )
        self.assertEqual(
            repo.get_targets("imu_data", "imu_1", IMU_TO_INSTRUMENT_SPEC),
            [("instrument_specification", "spec_1")],
        )
        self.assertEqual(
            repo.get_targets("feature", "feature_1", FEATURE_TO_IMU), [("imu_data", "imu_1")]
        )

    def test_imu_with_unindexed_user_aborts_build_without_writing(self) -> None:
        self._add_user("user_1")
        self._add_imu("imu_1", "user_1")
        self._add_imu("imu_2", "user_missing")

        with self.assertRaises(MetadataIndexError) as ctx:
            self._build()

        self.assertIn("imu_data:imu_2 -> user_data:user_missing", str(ctx.exception))
        self._assert_index_empty()

    def test_feature_with_unindexed_imu_aborts_build_without_writing(self) -> None:
        self._add_user("user_1")
        self._add_imu("imu_1", "user_1")
        self._add_feature("feature_1", "imu_missing")

        with self.assertRaises(MetadataIndexError) as ctx:
            self._build()

        self.assertIn("feature:feature_1 -> imu_data:imu_missing", str(ctx.exception))
        self._assert_index_empty()

    def test_all_missing_targets_are_reported_together(self) -> None:
        self._add_user("user_1")
        self._add_imu("imu_1", "user_a")
        self._add_imu("imu_2", "user_b")
        self._add_feature("feature_1", "imu_missing")

        with self.assertRaises(MetadataIndexError) as ctx:
            self._build()

        message = str(ctx.exception)
        for expected in (
            "imu_data:imu_1 -> user_data:user_a",
            "imu_data:imu_2 -> user_data:user_b",
            "feature:feature_1 -> imu_data:imu_missing",
        ):
            self.assertIn(expected, message)

    def test_rebuild_into_existing_index_updates_paths_and_keeps_relations(self) -> None:
        self._add_user("user_1")
        self._add_imu("imu_1", "user_1")
        self._add_feature("feature_1", "imu_1")
        self._build()

        repo = self._build()

        self.assertEqual(repo.list_record_ids("imu_data"), ["imu_1"])
        self.assertEqual(
            repo.get_sources("imu_data", "imu_1", FEATURE_TO_IMU), [("feature", "feature_1")]
        )


if __name__ == "__main__":
    unittest.main()
