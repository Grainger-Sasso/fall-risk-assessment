import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore


class _RepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "index.db"
        self.store = SQLiteStore(self.db_path)
        self.repo = MetadataRepository(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _index(self, *records: tuple[str, str]) -> None:
        for id_type, identifier in records:
            if not self.repo.record_exists(id_type, identifier):
                self.repo.upsert_record(
                    id_type, identifier, Path(self.temp_dir.name) / f"{id_type}_{identifier}"
                )

    def _relate_indexed(
        self,
        source: tuple[str, str],
        target: tuple[str, str],
        relation_type: str,
    ) -> None:
        """Index both endpoints if needed, then add the relation."""
        self._index(source, target)
        self.repo.add_relation(
            source_type=source[0],
            source_id=source[1],
            target_type=target[0],
            target_id=target[1],
            relation_type=relation_type,
        )

    def _count_rows(self, table: str) -> int:
        row = self.store.fetch_one(f"SELECT COUNT(*) AS n FROM {table};")
        return row["n"]


class TestSQLiteMetadataRepositoryRecords(_RepositoryTestCase):
    def test_upsert_and_get_record_path(self) -> None:
        record_path = Path(self.temp_dir.name) / "imu_data_123"
        self.repo.upsert_record("imu_data", "imu_1", record_path)

        fetched_path = self.repo.get_record_path("imu_data", "imu_1")
        self.assertEqual(fetched_path, record_path)
        self.assertTrue(self.repo.record_exists("imu_data", "imu_1"))
        self.assertEqual(self.repo.list_record_ids("imu_data"), ["imu_1"])

    def test_upsert_overwrites_existing_path(self) -> None:
        first_path = Path(self.temp_dir.name) / "first"
        second_path = Path(self.temp_dir.name) / "second"

        self.repo.upsert_record("imu_data", "imu_1", first_path)
        self.repo.upsert_record("imu_data", "imu_1", second_path)

        self.assertEqual(self.repo.get_record_path("imu_data", "imu_1"), second_path)
        self.assertEqual(self.repo.list_record_ids("imu_data"), ["imu_1"])
        self.assertEqual(self._count_rows("records"), 1)

    def test_upsert_preserves_created_at_and_advances_updated_at(self) -> None:
        timestamps = ["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"]
        with patch.object(self.store, "utc_now_iso", side_effect=timestamps):
            self.repo.upsert_record("imu_data", "imu_1", Path("a"))
            self.repo.upsert_record("imu_data", "imu_1", Path("b"))

        row = self.store.fetch_one(
            "SELECT created_at, updated_at FROM records WHERE id_type=? AND id=?;",
            ("imu_data", "imu_1"),
        )
        self.assertEqual(row["created_at"], timestamps[0])
        self.assertEqual(row["updated_at"], timestamps[1])

    def test_upsert_of_existing_record_keeps_its_relations(self) -> None:
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_to_user")

        self.repo.upsert_record("imu_data", "i1", Path(self.temp_dir.name) / "moved")

        self.assertEqual(self.repo.get_sources("imu_data", "i1"), [("feature", "f1")])
        self.assertEqual(self.repo.get_targets("imu_data", "i1"), [("user_data", "u1")])

    def test_get_record_path_raises_key_error_for_missing_record(self) -> None:
        with self.assertRaises(KeyError):
            self.repo.get_record_path("imu_data", "missing")

    def test_record_exists_is_false_for_unknown_id_or_mismatched_type(self) -> None:
        self.repo.upsert_record("imu_data", "imu_1", Path("imu_1"))

        self.assertFalse(self.repo.record_exists("imu_data", "missing"))
        self.assertFalse(self.repo.record_exists("feature", "imu_1"))

    def test_same_id_under_different_types_is_isolated(self) -> None:
        imu_path = Path(self.temp_dir.name) / "imu_x"
        feature_path = Path(self.temp_dir.name) / "feature_x"

        self.repo.upsert_record("imu_data", "x", imu_path)
        self.repo.upsert_record("feature", "x", feature_path)
        self.repo.upsert_record("feature", "y", feature_path)

        self.assertEqual(self.repo.get_record_path("imu_data", "x"), imu_path)
        self.assertEqual(self.repo.get_record_path("feature", "x"), feature_path)
        self.assertEqual(self.repo.list_record_ids("imu_data"), ["x"])
        self.assertEqual(self.repo.list_record_ids("feature"), ["x", "y"])

    def test_list_record_ids_is_sorted_and_empty_for_unused_type(self) -> None:
        for identifier in ["imu_c", "imu_a", "imu_b"]:
            self.repo.upsert_record("imu_data", identifier, Path(identifier))

        self.assertEqual(self.repo.list_record_ids("imu_data"), ["imu_a", "imu_b", "imu_c"])
        self.assertEqual(self.repo.list_record_ids("user_data"), [])

    def test_absolute_record_path_round_trips_unchanged(self) -> None:
        path = Path(self.temp_dir.name) / "dir with spaces" / "réçørd_ü"
        self.repo.upsert_record("imu_data", "imu_1", path)
        self.assertEqual(self.repo.get_record_path("imu_data", "imu_1"), path)

    def test_relative_record_path_is_stored_as_absolute(self) -> None:
        self.repo.upsert_record("imu_data", "imu_1", Path("relative") / "imu_1")

        stored = self.repo.get_record_path("imu_data", "imu_1")
        self.assertTrue(stored.is_absolute())
        self.assertEqual(stored, Path.cwd() / "relative" / "imu_1")

    def test_record_path_parent_segments_are_collapsed(self) -> None:
        root = Path(self.temp_dir.name)
        self.repo.upsert_record("imu_data", "imu_1", root / "a" / ".." / "b")
        self.assertEqual(self.repo.get_record_path("imu_data", "imu_1"), root / "b")

    def test_record_path_does_not_follow_symlinks(self) -> None:
        root = Path(self.temp_dir.name)
        (root / "real").mkdir()
        link = root / "link"
        os.symlink(root / "real", link)

        self.repo.upsert_record("imu_data", "imu_1", link)
        self.assertEqual(self.repo.get_record_path("imu_data", "imu_1"), link)


class TestSQLiteMetadataRepositoryRelations(_RepositoryTestCase):
    def test_add_and_query_relations(self) -> None:
        self._index(("feature", "feature_1"), ("imu_data", "imu_1"))
        self.repo.add_relation(
            source_type="feature",
            source_id="feature_1",
            target_type="imu_data",
            target_id="imu_1",
            relation_type="feature_to_imu",
        )
        targets = self.repo.get_targets("feature", "feature_1")
        self.assertEqual(targets, [("imu_data", "imu_1")])

        sources = self.repo.get_sources("imu_data", "imu_1")
        self.assertEqual(sources, [("feature", "feature_1")])

        self.assertTrue(
            self.repo.relation_exists(
                source_type="feature",
                source_id="feature_1",
                target_type="imu_data",
                target_id="imu_1",
                relation_type="feature_to_imu",
            )
        )

    def test_add_relation_rejects_unindexed_endpoints(self) -> None:
        cases = {
            "missing_source": [("imu_data", "i1")],
            "missing_target": [("feature", "f1")],
            "missing_both": [],
        }
        for case, indexed in cases.items():
            with self.subTest(case=case):
                self.store.execute("DELETE FROM records;")
                self._index(*indexed)
                with self.assertRaises(KeyError) as ctx:
                    self.repo.add_relation(
                        source_type="feature",
                        source_id="f1",
                        target_type="imu_data",
                        target_id="i1",
                        relation_type="feature_to_imu",
                    )
                message = str(ctx.exception)
                for id_type, identifier in [("feature", "f1"), ("imu_data", "i1")]:
                    expected_missing = (id_type, identifier) not in indexed
                    self.assertEqual(f"{id_type}:{identifier}" in message, expected_missing)
                self.assertEqual(self._count_rows("relations"), 0)

    def test_schema_rejects_relations_to_unindexed_records(self) -> None:
        self._index(("feature", "f1"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.execute(
                "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?);",
                ("f1", "feature", "i1", "imu_data", "feature_to_imu", "ts"),
            )

    def test_duplicate_relation_is_ignored(self) -> None:
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")

        self.assertEqual(self.repo.get_targets("feature", "f1"), [("imu_data", "i1")])
        self.assertEqual(self._count_rows("relations"), 1)

    def test_same_pair_with_different_relation_types_stores_both(self) -> None:
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_to_user")
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_owned_by")

        self.assertEqual(self._count_rows("relations"), 2)
        self.assertEqual(
            self.repo.get_targets("imu_data", "i1", relation_type="imu_to_user"),
            [("user_data", "u1")],
        )
        self.assertEqual(
            self.repo.get_sources("user_data", "u1", relation_type="imu_owned_by"),
            [("imu_data", "i1")],
        )

    def test_relation_type_filter_on_targets_and_sources(self) -> None:
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_to_user")
        self._relate_indexed(
            ("imu_data", "i1"), ("instrument_specification", "s1"), "imu_to_instrument_spec"
        )
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")
        self._relate_indexed(("feature", "f2"), ("imu_data", "i1"), "derived_from")

        self.assertEqual(
            self.repo.get_targets("imu_data", "i1", relation_type="imu_to_user"),
            [("user_data", "u1")],
        )
        self.assertEqual(
            self.repo.get_targets("imu_data", "i1", relation_type="imu_to_instrument_spec"),
            [("instrument_specification", "s1")],
        )
        self.assertEqual(
            self.repo.get_sources("imu_data", "i1", relation_type="feature_to_imu"),
            [("feature", "f1")],
        )
        self.assertEqual(
            self.repo.get_targets("imu_data", "i1", relation_type="nonexistent"), []
        )
        self.assertEqual(
            self.repo.get_sources("imu_data", "i1", relation_type="nonexistent"), []
        )

    def test_targets_and_sources_are_sorted_by_type_then_id(self) -> None:
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u2"), "r")
        self._relate_indexed(("imu_data", "i1"), ("instrument_specification", "s1"), "r")
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "r")
        self._relate_indexed(("user_data", "u9"), ("imu_data", "target"), "r")
        self._relate_indexed(("feature", "f2"), ("imu_data", "target"), "r")
        self._relate_indexed(("feature", "f1"), ("imu_data", "target"), "r")

        self.assertEqual(
            self.repo.get_targets("imu_data", "i1"),
            [("instrument_specification", "s1"), ("user_data", "u1"), ("user_data", "u2")],
        )
        self.assertEqual(
            self.repo.get_sources("imu_data", "target"),
            [("feature", "f1"), ("feature", "f2"), ("user_data", "u9")],
        )

    def test_relation_exists_requires_exact_match_on_all_fields(self) -> None:
        relation = {
            "source_type": "feature",
            "source_id": "f1",
            "target_type": "imu_data",
            "target_id": "i1",
            "relation_type": "feature_to_imu",
        }
        self._index(("feature", "f1"), ("imu_data", "i1"))
        self.repo.add_relation(**relation)
        self.assertTrue(self.repo.relation_exists(**relation))

        for field in relation:
            with self.subTest(mismatched_field=field):
                mismatched = {**relation, field: "other"}
                self.assertFalse(self.repo.relation_exists(**mismatched))

    def test_relation_exists_without_relation_type_matches_any_type(self) -> None:
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_to_user")

        self.assertTrue(
            self.repo.relation_exists(
                source_type="imu_data",
                source_id="i1",
                target_type="user_data",
                target_id="u1",
            )
        )
        self.assertFalse(
            self.repo.relation_exists(
                source_type="imu_data",
                source_id="i1",
                target_type="user_data",
                target_id="other",
            )
        )

    def test_relations_are_directional(self) -> None:
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")

        self.assertEqual(self.repo.get_targets("imu_data", "i1"), [])
        self.assertEqual(self.repo.get_sources("feature", "f1"), [])
        self.assertFalse(
            self.repo.relation_exists(
                source_type="imu_data",
                source_id="i1",
                target_type="feature",
                target_id="f1",
                relation_type="feature_to_imu",
            )
        )


class TestSQLiteMetadataRepositoryDeletes(_RepositoryTestCase):
    def test_delete_record_removes_only_matching_type_and_id(self) -> None:
        self.repo.upsert_record("feature", "x", Path("feature_x"))
        self.repo.upsert_record("imu_data", "x", Path("imu_x"))
        self.repo.upsert_record("feature", "y", Path("feature_y"))

        self.repo.delete_record("feature", "x")

        self.assertFalse(self.repo.record_exists("feature", "x"))
        self.assertTrue(self.repo.record_exists("imu_data", "x"))
        self.assertTrue(self.repo.record_exists("feature", "y"))

    def test_delete_record_cascades_to_relations_in_both_directions(self) -> None:
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_to_user")
        self._relate_indexed(("imu_data", "i2"), ("user_data", "u1"), "imu_to_user")

        self.repo.delete_record("imu_data", "i1")

        self.assertEqual(self.repo.get_targets("feature", "f1"), [])
        self.assertEqual(self.repo.get_sources("user_data", "u1"), [("imu_data", "i2")])
        self.assertTrue(self.repo.record_exists("feature", "f1"))
        self.assertTrue(self.repo.record_exists("user_data", "u1"))

    def test_delete_relations_by_source_keeps_relations_where_node_is_target(self) -> None:
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_to_user")
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")
        self._relate_indexed(("imu_data", "i2"), ("user_data", "u1"), "imu_to_user")

        self.repo.delete_relations_by_source("imu_data", "i1")

        self.assertEqual(self.repo.get_targets("imu_data", "i1"), [])
        self.assertEqual(self.repo.get_sources("imu_data", "i1"), [("feature", "f1")])
        self.assertEqual(self.repo.get_targets("imu_data", "i2"), [("user_data", "u1")])
        self.assertTrue(self.repo.record_exists("imu_data", "i1"))

    def test_delete_relations_by_target_keeps_relations_where_node_is_source(self) -> None:
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")
        self._relate_indexed(("imu_data", "i1"), ("user_data", "u1"), "imu_to_user")
        self._relate_indexed(("feature", "f2"), ("imu_data", "i2"), "feature_to_imu")

        self.repo.delete_relations_by_target("imu_data", "i1")

        self.assertEqual(self.repo.get_sources("imu_data", "i1"), [])
        self.assertEqual(self.repo.get_targets("imu_data", "i1"), [("user_data", "u1")])
        self.assertEqual(self.repo.get_sources("imu_data", "i2"), [("feature", "f2")])
        self.assertTrue(self.repo.record_exists("imu_data", "i1"))

    def test_deleting_nonexistent_entries_is_a_no_op(self) -> None:
        self._relate_indexed(("feature", "f1"), ("imu_data", "i1"), "feature_to_imu")

        self.repo.delete_record("imu_data", "missing")
        self.repo.delete_relations_by_source("feature", "missing")
        self.repo.delete_relations_by_target("imu_data", "missing")

        self.assertEqual(self._count_rows("records"), 2)
        self.assertEqual(self._count_rows("relations"), 1)


class TestSQLiteMetadataRepositoryValidation(_RepositoryTestCase):
    def test_public_methods_reject_empty_identifiers_and_types(self) -> None:
        relation_args = ("feature", "f1", "imu_data", "i1", "feature_to_imu")
        calls = {
            "upsert_record": lambda bad: self.repo.upsert_record(bad, "id", Path("p")),
            "upsert_record_id": lambda bad: self.repo.upsert_record("imu_data", bad, Path("p")),
            "get_record_path": lambda bad: self.repo.get_record_path("imu_data", bad),
            "record_exists": lambda bad: self.repo.record_exists(bad, "id"),
            "list_record_ids": lambda bad: self.repo.list_record_ids(bad),
            "add_relation_source": lambda bad: self.repo.add_relation(bad, *relation_args[1:]),
            "add_relation_type": lambda bad: self.repo.add_relation(*relation_args[:4], bad),
            "get_targets": lambda bad: self.repo.get_targets("feature", bad),
            "get_targets_relation": lambda bad: self.repo.get_targets("feature", "f1", bad),
            "get_sources": lambda bad: self.repo.get_sources(bad, "i1"),
            "get_sources_relation": lambda bad: self.repo.get_sources("imu_data", "i1", bad),
            "relation_exists": lambda bad: self.repo.relation_exists("feature", "f1", bad, "i1"),
            "relation_exists_relation": lambda bad: self.repo.relation_exists(
                *relation_args[:4], bad
            ),
            "delete_record": lambda bad: self.repo.delete_record("imu_data", bad),
            "delete_relations_by_source": lambda bad: self.repo.delete_relations_by_source(
                bad, "f1"
            ),
            "delete_relations_by_target": lambda bad: self.repo.delete_relations_by_target(
                "imu_data", bad
            ),
        }
        for name, call in calls.items():
            for bad in ["", "   "]:
                with self.subTest(method=name, value=bad):
                    with self.assertRaises(ValueError):
                        call(bad)
        self.assertEqual(self._count_rows("records"), 0)


class TestSQLiteMetadataRepositoryTransactions(_RepositoryTestCase):
    def test_grouped_writes_commit_together(self) -> None:
        with self.repo.transaction():
            self.repo.upsert_record("feature", "f1", Path("f1"))
            self.repo.upsert_record("imu_data", "i1", Path("i1"))
            self.repo.add_relation("feature", "f1", "imu_data", "i1", "feature_to_imu")

        reopened = MetadataRepository(SQLiteStore(self.db_path))
        self.assertTrue(reopened.relation_exists("feature", "f1", "imu_data", "i1"))

    def test_grouped_writes_roll_back_together_on_failure(self) -> None:
        with self.assertRaises(KeyError):
            with self.repo.transaction():
                self.repo.upsert_record("feature", "f1", Path("f1"))
                self.repo.add_relation("feature", "f1", "imu_data", "i1", "feature_to_imu")

        self.assertFalse(self.repo.record_exists("feature", "f1"))
        self.assertEqual(self._count_rows("records"), 0)


class TestSQLiteMetadataRepositoryPersistence(_RepositoryTestCase):
    def test_data_survives_reopening_store(self) -> None:
        record_path = Path(self.temp_dir.name) / "imu_1"
        self.repo.upsert_record("imu_data", "imu_1", record_path)
        self._relate_indexed(("imu_data", "imu_1"), ("user_data", "u1"), "imu_to_user")

        reopened = MetadataRepository(SQLiteStore(self.db_path))

        self.assertEqual(reopened.get_record_path("imu_data", "imu_1"), record_path)
        self.assertEqual(reopened.get_targets("imu_data", "imu_1"), [("user_data", "u1")])
        self.assertEqual(self._count_rows("schema_version"), 1)


if __name__ == "__main__":
    unittest.main()
