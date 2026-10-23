import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable, Optional

from src.database_manager.database_errors import SchemaVersionError
from src.database_manager.metadata.sql.ddl import DDL_STATEMENTS, SCHEMA_VERSION


class SQLiteStore:
    """Thin sqlite3 wrapper for metadata persistence.

    ``transaction()`` is re-entrant: nested calls share the outermost connection
    and run as savepoints, so only the outermost block commits. Reads issued while
    a transaction is open use that connection and see its uncommitted writes. The
    open-transaction state is per instance and not thread-safe.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_conn: Optional[sqlite3.Connection] = None
        self._savepoint_depth = 0
        self._initialize_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        if self._active_conn is not None:
            with self._savepoint(self._active_conn) as conn:
                yield conn
            return

        conn = self.connect()
        self._active_conn = conn
        try:
            conn.execute("BEGIN;")
            yield conn
            conn.execute("COMMIT;")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            self._active_conn = None
            conn.close()

    @contextmanager
    def _savepoint(
        self, conn: sqlite3.Connection
    ) -> Generator[sqlite3.Connection, None, None]:
        self._savepoint_depth += 1
        name = f"sp_{self._savepoint_depth}"
        conn.execute(f"SAVEPOINT {name};")
        try:
            yield conn
        except BaseException:
            conn.execute(f"ROLLBACK TO {name};")
            conn.execute(f"RELEASE {name};")
            raise
        else:
            conn.execute(f"RELEASE {name};")
        finally:
            self._savepoint_depth -= 1

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.transaction() as conn:
            conn.execute(sql, params)

    def execute_many(self, sql: str, param_rows: Iterable[tuple]) -> None:
        with self.transaction() as conn:
            conn.executemany(sql, param_rows)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        if self._active_conn is not None:
            return self._active_conn.execute(sql, params).fetchone()
        conn = self.connect()
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        if self._active_conn is not None:
            return self._active_conn.execute(sql, params).fetchall()
        conn = self.connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self.transaction() as conn:
            existing = self._read_schema_version(conn)
            if existing is not None and existing != SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"SQLite index {self.db_path} uses schema version {existing}; "
                    f"expected {SCHEMA_VERSION}. Regenerate the index."
                )
            for statement in DDL_STATEMENTS:
                conn.execute(statement)
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?);",
                    (SCHEMA_VERSION,),
                )

    @staticmethod
    def _read_schema_version(conn: sqlite3.Connection) -> Optional[str]:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version';"
        ).fetchone()
        if table is None:
            return None
        row = conn.execute("SELECT version FROM schema_version LIMIT 1;").fetchone()
        return None if row is None else row["version"]

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
