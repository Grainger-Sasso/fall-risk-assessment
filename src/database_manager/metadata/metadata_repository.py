import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from src.database_manager.metadata.sqlite_store import SQLiteStore


class MetadataRepository:
    """Repository for metadata index operations on records and relations.

    Relations require both endpoints to be indexed records; deleting a record
    cascades to every relation that references it.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Group repository calls so they commit or roll back together."""
        with self.store.transaction():
            yield

    def upsert_record(self, id_type: str, identifier: str, path: Path) -> None:
        self._require_non_empty(id_type=id_type, identifier=identifier)
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
                (identifier, id_type, os.path.abspath(path), ts, ts),
            )

    def get_record_path(self, id_type: str, identifier: str) -> Path:
        self._require_non_empty(id_type=id_type, identifier=identifier)
        row = self.store.fetch_one(
            "SELECT path FROM records WHERE id_type=? AND id=?;",
            (id_type, identifier),
        )
        if row is None:
            raise KeyError(f"No path found for {id_type}:{identifier}")
        return Path(row["path"])

    def record_exists(self, id_type: str, identifier: str) -> bool:
        self._require_non_empty(id_type=id_type, identifier=identifier)
        row = self.store.fetch_one(
            "SELECT 1 FROM records WHERE id_type=? AND id=? LIMIT 1;",
            (id_type, identifier),
        )
        return row is not None

    def list_record_ids(self, id_type: str) -> list[str]:
        self._require_non_empty(id_type=id_type)
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
        self._require_non_empty(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relation_type=relation_type,
        )
        ts = self.store.utc_now_iso()
        with self.store.transaction() as conn:
            missing = [
                f"{id_type}:{identifier}"
                for id_type, identifier in (
                    (source_type, source_id),
                    (target_type, target_id),
                )
                if not self.record_exists(id_type, identifier)
            ]
            if missing:
                raise KeyError(
                    f"Cannot add relation {relation_type}; records not indexed: "
                    + ", ".join(missing)
                )
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
        self._require_non_empty(source_type=source_type, source_id=source_id)
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
            self._require_non_empty(relation_type=relation_type)
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
        self._require_non_empty(target_type=target_type, target_id=target_id)
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
            self._require_non_empty(relation_type=relation_type)
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
        relation_type: Optional[str] = None,
    ) -> bool:
        self._require_non_empty(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
        )
        sql = """
            SELECT 1
            FROM relations
            WHERE source_type=? AND source_id=? AND target_type=? AND target_id=?
        """
        params: tuple = (source_type, source_id, target_type, target_id)
        if relation_type is not None:
            self._require_non_empty(relation_type=relation_type)
            sql += " AND relation_type=?"
            params += (relation_type,)
        row = self.store.fetch_one(sql + " LIMIT 1;", params)
        return row is not None

    def delete_record(self, id_type: str, identifier: str) -> None:
        self._require_non_empty(id_type=id_type, identifier=identifier)
        with self.store.transaction() as conn:
            conn.execute(
                "DELETE FROM records WHERE id_type=? AND id=?;",
                (id_type, identifier),
            )

    def delete_relations_by_source(self, source_type: str, source_id: str) -> None:
        self._require_non_empty(source_type=source_type, source_id=source_id)
        with self.store.transaction() as conn:
            conn.execute(
                "DELETE FROM relations WHERE source_type=? AND source_id=?;",
                (source_type, source_id),
            )

    def delete_relations_by_target(self, target_type: str, target_id: str) -> None:
        self._require_non_empty(target_type=target_type, target_id=target_id)
        with self.store.transaction() as conn:
            conn.execute(
                "DELETE FROM relations WHERE target_type=? AND target_id=?;",
                (target_type, target_id),
            )

    @staticmethod
    def _require_non_empty(**values: str) -> None:
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
