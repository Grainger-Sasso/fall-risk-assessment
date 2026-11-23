import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path, PureWindowsPath

from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.database_manager.relocate_index import (
    find_indexes,
    main,
    rebase_path,
    relocate_index,
)

OLD_ROOT = "E:/fafra_data/assessment_database"


def _paths(index: Path) -> dict:
    with closing(sqlite3.connect(index)) as conn:
        return {row[0]: row[1] for row in conn.execute("SELECT id, path FROM records")}


class TestRebasePath(unittest.TestCase):
    def test_windows_roots_match_either_separator_and_drive_case(self):
        new = "C:/fafra_data/assessment_database"
        expected = str(PureWindowsPath(new) / "imu_data" / "ltmm" / "imu_data_x")
        for stored in (
            r"E:\fafra_data\assessment_database\imu_data\ltmm\imu_data_x",
            "E:/fafra_data/assessment_database/imu_data/ltmm/imu_data_x",
            "e:/fafra_data/assessment_database/imu_data/ltmm/imu_data_x/",
        ):
            self.assertEqual(rebase_path(stored, OLD_ROOT, new), expected, stored)

    def test_paths_outside_the_root_are_not_rebased(self):
        new = "C:/db"
        self.assertIsNone(rebase_path("/Volumes/Backup Plus/fafra_data/assessment_database/x", OLD_ROOT, new))
        self.assertIsNone(rebase_path("E:/fafra_data/assessment_database_old/x", OLD_ROOT, new))
        self.assertEqual(rebase_path(OLD_ROOT, OLD_ROOT, new), str(PureWindowsPath(new)))

    def test_posix_roots_are_case_sensitive(self):
        old = "/Volumes/Backup Plus/fafra_data/assessment_database"
        self.assertEqual(rebase_path(old + "/user_data/u1", old, "/data/db"), "/data/db/user_data/u1")
        self.assertIsNone(rebase_path(old.lower() + "/user_data/u1", old, "/data/db"))


class TestRelocateIndex(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.new_root = self.tmp / "assessment_database"
        for rel in ("imu_data/ds/imu_data_a", "user_data/ds/user_u1"):
            (self.new_root / rel).mkdir(parents=True)
        self.index = self.tmp / "sql_db_indexes" / "ds" / "index.db"
        SQLiteStore(self.index)  # creates the schema
        rows = [
            ("imu_a", "IMUDataIdentifier", r"E:\fafra_data\assessment_database\imu_data\ds\imu_data_a"),
            ("u1", "UserIdentifier", "E:/fafra_data/assessment_database/user_data/ds/user_u1"),
            ("u2", "UserIdentifier", "/Volumes/Backup Plus/fafra_data/assessment_database/user_data/ds/user_u2"),
            ("u3", "UserIdentifier", str(self.new_root / "user_data" / "ds" / "user_u1")),
        ]
        with closing(sqlite3.connect(self.index)) as conn, conn:
            conn.executemany(
                "INSERT INTO records(id, id_type, path, created_at, updated_at) VALUES (?, ?, ?, 't', 't')",
                rows,
            )

    def tearDown(self):
        self._tmp.cleanup()

    def test_rewrites_matching_rows_and_reports_the_rest(self):
        report = relocate_index(self.index, OLD_ROOT, str(self.new_root))
        self.assertTrue(report.written)
        self.assertEqual(report.rewritten, 2)
        self.assertEqual(report.already_under_new_root, 1)
        self.assertEqual(sum(report.other_roots.values()), 1)
        self.assertEqual(report.missing, [])

        paths = _paths(self.index)
        self.assertEqual(Path(paths["imu_a"]), self.new_root / "imu_data" / "ds" / "imu_data_a")
        self.assertEqual(Path(paths["u1"]), self.new_root / "user_data" / "ds" / "user_u1")
        self.assertTrue(paths["u2"].startswith("/Volumes/"))

        repository = MetadataRepository(SQLiteStore(self.index))
        self.assertTrue(repository.get_record_path("IMUDataIdentifier", "imu_a").is_dir())

    def test_dry_run_writes_nothing(self):
        before = _paths(self.index)
        report = relocate_index(self.index, OLD_ROOT, str(self.new_root), dry_run=True)
        self.assertEqual(report.rewritten, 2)
        self.assertFalse(report.written)
        self.assertEqual(_paths(self.index), before)

    def test_missing_targets_abort_without_writing(self):
        (self.new_root / "user_data" / "ds" / "user_u1").rmdir()
        before = _paths(self.index)
        with self.assertRaises(FileNotFoundError):
            relocate_index(self.index, OLD_ROOT, str(self.new_root))
        self.assertEqual(_paths(self.index), before)

        report = relocate_index(self.index, OLD_ROOT, str(self.new_root), allow_missing=True)
        self.assertTrue(report.written)
        self.assertEqual(len(report.missing), 1)

    def test_cli_expands_folders_and_ignores_macos_resource_forks(self):
        (self.index.parent / "._index.db").write_bytes(b"\x00\x05\x16\x07 not sqlite")
        self.assertEqual(find_indexes([self.tmp / "sql_db_indexes"]), [self.index])
        main(["--from", OLD_ROOT, "--to", str(self.new_root), str(self.tmp / "sql_db_indexes")])
        self.assertEqual(Path(_paths(self.index)["u1"]), self.new_root / "user_data" / "ds" / "user_u1")


if __name__ == "__main__":
    unittest.main()
