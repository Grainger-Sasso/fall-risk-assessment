from pathlib import Path
from typing import Type

from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.type_registry import IdentifierTypeRegistry
from src.identifiers.identifier import Identifier


class RegistryManager:
    """Registry facade backed by SQLite records table."""

    def __init__(self, metadata_repository: MetadataRepository):
        self.repository = metadata_repository

    def get_path(self, identifier: Identifier) -> Path:
        id_type = IdentifierTypeRegistry.get_type_name(type(identifier))
        return self.repository.get_record_path(id_type, identifier.value)

    def upsert_path(self, identifier: Identifier, path: Path) -> None:
        id_type = IdentifierTypeRegistry.get_type_name(type(identifier))
        self.repository.upsert_record(id_type, identifier.value, path)

    def exists(self, identifier: Identifier) -> bool:
        id_type = IdentifierTypeRegistry.get_type_name(type(identifier))
        return self.repository.record_exists(id_type, identifier.value)

    def list_ids(self, id_type: Type[Identifier]) -> list[Identifier]:
        id_type_name = IdentifierTypeRegistry.get_type_name(id_type)
        return [id_type(value) for value in self.repository.list_record_ids(id_type_name)]
