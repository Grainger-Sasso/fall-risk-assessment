SCHEMA_VERSION = "1.0"

CREATE_SCHEMA_VERSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY
);
"""

CREATE_RECORDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id TEXT NOT NULL,
    id_type TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (id_type, id)
);
"""

CREATE_RELATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS relations (
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_type, source_id, relation_type, target_type, target_id)
);
"""

CREATE_RECORDS_PATH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_records_path ON records(path);
"""

CREATE_RELATIONS_SOURCE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_relations_source
ON relations(source_type, source_id);
"""

CREATE_RELATIONS_TARGET_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_relations_target
ON relations(target_type, target_id);
"""

CREATE_RELATIONS_TYPE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_relations_type
ON relations(relation_type);
"""

DDL_STATEMENTS = [
    CREATE_SCHEMA_VERSION_TABLE_SQL,
    CREATE_RECORDS_TABLE_SQL,
    CREATE_RELATIONS_TABLE_SQL,
    CREATE_RECORDS_PATH_INDEX_SQL,
    CREATE_RELATIONS_SOURCE_INDEX_SQL,
    CREATE_RELATIONS_TARGET_INDEX_SQL,
    CREATE_RELATIONS_TYPE_INDEX_SQL,
]
