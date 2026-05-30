from typing import Optional

from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.type_registry import IdentifierTypeRegistry
from src.identifiers.identifier import Identifier


class MappingManager:
    """Mapping facade backed by SQLite relations table."""

    def __init__(self, metadata_repository: MetadataRepository):
        self.repository = metadata_repository

    def add_relation(
        self,
        source_id: Identifier,
        target_id: Identifier,
        relation_type: Optional[str] = None,
    ) -> None:
        source_type = IdentifierTypeRegistry.get_type_name(type(source_id))
        target_type = IdentifierTypeRegistry.get_type_name(type(target_id))
        relation_name = relation_type or IdentifierTypeRegistry.infer_relation_type(
            source_type, target_type
        )
        self.repository.add_relation(
            source_type=source_type,
            source_id=source_id.value,
            target_type=target_type,
            target_id=target_id.value,
            relation_type=relation_name,
        )

    def get_targets(
        self, source_id: Identifier, relation_type: Optional[str] = None
    ) -> list[Identifier]:
        source_type = IdentifierTypeRegistry.get_type_name(type(source_id))
        targets = self.repository.get_targets(
            source_type=source_type,
            source_id=source_id.value,
            relation_type=relation_type,
        )
        output = []
        for target_type_name, target_id_value in targets:
            try:
                output.append(
                    IdentifierTypeRegistry.build_identifier(target_type_name, target_id_value)
                )
            except KeyError:
                # Ignore unknown identifier types to keep index resilient to partial migrations.
                continue
        return output

    def get_single_target(
        self, source_id: Identifier, relation_type: Optional[str] = None
    ) -> Optional[Identifier]:
        targets = self.get_targets(source_id=source_id, relation_type=relation_type)
        if not targets:
            return None
        return targets[0]
