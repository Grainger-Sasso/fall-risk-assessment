from pathlib import Path
from typing import Optional

from src.database_manager.metadata.sqlite_store import SQLiteStore


class MetadataRepository:
    """Repository for metadata index operations on records and relations."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def upsert_record(self, id_type: str, identifier: str, path: Path) -> None:
        ts = self.store.utc_now_iso()
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO records(id, id_type, path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id_type, id) DO UPDATE SET
                    path=excluded.path,
                    updated_at=excluded.updated_at;
                """,
                (identifier, id_type, str(path), ts, ts),
            )

    def get_record_path(self, id_type: str, identifier: str) -> Path:
        row = self.store.fetch_one(
            "SELECT path FROM records WHERE id_type=? AND id=?;",
            (id_type, identifier),
        )
        if row is None:
            raise KeyError(f"No path found for {id_type}:{identifier}")
        return Path(row["path"])

    def record_exists(self, id_type: str, identifier: str) -> bool:
        row = self.store.fetch_one(
            "SELECT 1 FROM records WHERE id_type=? AND id=? LIMIT 1;",
            (id_type, identifier),
        )
        return row is not None

    def list_record_ids(self, id_type: str) -> list[str]:
        rows = self.store.fetch_all(
            "SELECT id FROM records WHERE id_type=? ORDER BY id;",
            (id_type,),
        )
        return [row["id"] for row in rows]

    def add_relation(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation_type: str,
    ) -> None:
        ts = self.store.utc_now_iso()
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO relations(
                    source_id, source_type, target_id, target_type, relation_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (source_id, source_type, target_id, target_type, relation_type, ts),
            )

    def get_targets(
        self, source_type: str, source_id: str, relation_type: Optional[str] = None
    ) -> list[tuple[str, str]]:
        if relation_type is None:
            rows = self.store.fetch_all(
                """
                SELECT target_type, target_id
                FROM relations
                WHERE source_type=? AND source_id=?
                ORDER BY target_type, target_id;
                """,
                (source_type, source_id),
            )
        else:
            rows = self.store.fetch_all(
                """
                SELECT target_type, target_id
                FROM relations
                WHERE source_type=? AND source_id=? AND relation_type=?
                ORDER BY target_type, target_id;
                """,
                (source_type, source_id, relation_type),
            )
        return [(row["target_type"], row["target_id"]) for row in rows]

    def get_sources(
        self, target_type: str, target_id: str, relation_type: Optional[str] = None
    ) -> list[tuple[str, str]]:
        if relation_type is None:
            rows = self.store.fetch_all(
                """
                SELECT source_type, source_id
                FROM relations
                WHERE target_type=? AND target_id=?
                ORDER BY source_type, source_id;
                """,
                (target_type, target_id),
            )
        else:
            rows = self.store.fetch_all(
                """
                SELECT source_type, source_id
                FROM relations
                WHERE target_type=? AND target_id=? AND relation_type=?
                ORDER BY source_type, source_id;
                """,
                (target_type, target_id, relation_type),
            )
        return [(row["source_type"], row["source_id"]) for row in rows]

    def relation_exists(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation_type: str,
    ) -> bool:
        row = self.store.fetch_one(
            """
            SELECT 1
            FROM relations
            WHERE source_type=? AND source_id=? AND target_type=? AND target_id=?
                AND relation_type=?
            LIMIT 1;
            """,
            (source_type, source_id, target_type, target_id, relation_type),
        )
        return row is not None
