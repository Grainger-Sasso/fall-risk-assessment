import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.database_manager.database_errors import MetadataIndexError, SchemaVersionError
from src.database_manager.metadata.sql.ddl import SCHEMA_VERSION
from src.database_manager.metadata.sqlite_store import SQLiteStore


class _StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "nested" / "index.db"
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _insert_record(self, identifier: str) -> None:
        self.store.execute(
            "INSERT INTO records(id, id_type, path, created_at, updated_at) "
            "VALUES (?, 'imu_data', 'p', 't', 't');",
            (identifier,),
        )

    def _record_ids(self, store: SQLiteStore) -> list[str]:
        return [row["id"] for row in store.fetch_all("SELECT id FROM records ORDER BY id;")]


class TestSQLiteStoreSchema(_StoreTestCase):
    def test_new_database_creates_parent_dir_and_records_schema_version(self) -> None:
        self.assertTrue(self.db_path.exists())
        rows = self.store.fetch_all("SELECT version FROM schema_version;")
        self.assertEqual([row["version"] for row in rows], [SCHEMA_VERSION])

    def test_reopening_does_not_duplicate_schema_version_or_drop_data(self) -> None:
        self._insert_record("i1")

        reopened = SQLiteStore(self.db_path)

        self.assertEqual(len(reopened.fetch_all("SELECT version FROM schema_version;")), 1)
        self.assertEqual(self._record_ids(reopened), ["i1"])

    def test_mismatched_schema_version_raises_and_leaves_index_untouched(self) -> None:
        self.store.execute("UPDATE schema_version SET version='0.9';")

        with self.assertRaises(SchemaVersionError) as ctx:
            SQLiteStore(self.db_path)

        self.assertIsInstance(ctx.exception, MetadataIndexError)
        self.assertIn("0.9", str(ctx.exception))
        self.assertIn("Regenerate", str(ctx.exception))
        row = self.store.fetch_one("SELECT version FROM schema_version;")
        self.assertEqual(row["version"], "0.9")

    def test_connections_enforce_foreign_keys(self) -> None:
        conn = self.store.connect()
        try:
            self.assertEqual(conn.execute("PRAGMA foreign_keys;").fetchone()[0], 1)
        finally:
            conn.close()


class TestSQLiteStoreTransactions(_StoreTestCase):
    def test_transaction_commits_on_success(self) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO records VALUES ('i1', 'imu_data', 'p', 't', 't');"
            )

        self.assertEqual(self._record_ids(SQLiteStore(self.db_path)), ["i1"])

    def test_transaction_rolls_back_on_error(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.store.transaction():
                self._insert_record("i1")
                raise RuntimeError("boom")

        self.assertEqual(self._record_ids(self.store), [])

    def test_nested_writes_commit_only_with_outermost_transaction(self) -> None:
        observer = SQLiteStore(self.db_path)
        with self.store.transaction():
            self._insert_record("i1")
            with self.store.transaction():
                self._insert_record("i2")
            self.assertEqual(self._record_ids(observer), [])

        self.assertEqual(self._record_ids(observer), ["i1", "i2"])

    def test_outer_failure_rolls_back_completed_inner_transactions(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.store.transaction():
                with self.store.transaction():
                    self._insert_record("i1")
                with self.store.transaction():
                    with self.store.transaction():
                        self._insert_record("i2")
                raise RuntimeError("boom")

        self.assertEqual(self._record_ids(self.store), [])

    def test_caught_inner_failure_rolls_back_only_inner_writes(self) -> None:
        with self.store.transaction():
            self._insert_record("outer")
            try:
                with self.store.transaction():
                    self._insert_record("inner")
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            self._insert_record("after")

        self.assertEqual(self._record_ids(self.store), ["after", "outer"])

    def test_reads_inside_transaction_see_uncommitted_writes(self) -> None:
        with self.store.transaction():
            self._insert_record("i1")
            self.assertEqual(self._record_ids(self.store), ["i1"])
            row = self.store.fetch_one("SELECT id FROM records WHERE id='i1';")
            self.assertIsNotNone(row)

    def test_store_is_reusable_after_failed_transaction(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.transaction():
                self._insert_record("dup")
                self._insert_record("dup")

        self._insert_record("i1")
        with self.store.transaction():
            self._insert_record("i2")

        self.assertEqual(self._record_ids(SQLiteStore(self.db_path)), ["i1", "i2"])


if __name__ == "__main__":
    unittest.main()
