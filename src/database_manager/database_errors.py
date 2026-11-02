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


class DatabaseValidationError(DatabaseManagerError):
    """Raised when indexed records disagree with their payloads; lists every issue found."""

    def __init__(self, scope: str, issues: list[str]) -> None:
        self.scope = scope
        self.issues = issues
        details = "\n".join(f"  - {issue}" for issue in issues)
        super().__init__(f"{scope} validation found {len(issues)} issue(s):\n{details}")
