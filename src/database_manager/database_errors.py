class DatabaseManagerError(Exception):
    """Base exception type for database manager operations."""


class DatabaseWriteError(DatabaseManagerError):
    """Raised when writing payloads or metadata fails."""


class DatabaseReadError(DatabaseManagerError):
    """Raised when loading payloads or metadata fails."""


class MetadataIndexError(DatabaseManagerError):
    """Raised when SQLite metadata index operations fail."""


class SchemaVersionError(MetadataIndexError):
    """Raised when an existing SQLite index was built with a different schema version."""
