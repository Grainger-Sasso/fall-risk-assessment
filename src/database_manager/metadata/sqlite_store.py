import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable, Optional

from src.database_manager.metadata.sql.ddl import DDL_STATEMENTS, SCHEMA_VERSION


class SQLiteStore:
    """Thin sqlite3 wrapper for metadata persistence."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.transaction() as conn:
            conn.execute(sql, params)

    def execute_many(self, sql: str, param_rows: Iterable[tuple]) -> None:
        with self.transaction() as conn:
            conn.executemany(sql, param_rows)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        conn = self.connect()
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self.connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self.transaction() as conn:
            for statement in DDL_STATEMENTS:
                conn.execute(statement)
            existing = conn.execute("SELECT version FROM schema_version LIMIT 1;").fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?);",
                    (SCHEMA_VERSION,),
                )

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
