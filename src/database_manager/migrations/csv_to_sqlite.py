import csv
from pathlib import Path
from typing import Dict, Tuple, Type

from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.type_registry import IdentifierTypeRegistry
from src.identifiers.identifier import Identifier


def _read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Expected CSV file not found: {file_path}")
    with file_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def migrate_csv_indexes(
    repository: MetadataRepository,
    registry_paths: Dict[Type[Identifier], Path],
    mapping_paths: Dict[Tuple[Type[Identifier], Type[Identifier]], Path],
) -> None:
    """Populate SQLite index using legacy registry/mapping CSV folders."""
    for id_type, subdir in registry_paths.items():
        csv_path = subdir / "registry.csv"
        rows = _read_csv_rows(csv_path)
        id_type_name = IdentifierTypeRegistry.get_type_name(id_type)
        for row in rows:
            identifier = row["data_identifier"]
            directory = Path(row["directory"])
            repository.upsert_record(
                id_type=id_type_name,
                identifier=identifier,
                path=directory,
            )

    for (source_type, target_type), subdir in mapping_paths.items():
        csv_path = subdir / "mapping.csv"
        rows = _read_csv_rows(csv_path)
        source_type_name = IdentifierTypeRegistry.get_type_name(source_type)
        target_type_name = IdentifierTypeRegistry.get_type_name(target_type)
        relation_type = IdentifierTypeRegistry.infer_relation_type(
            source_type_name, target_type_name
        )
        for row in rows:
            repository.add_relation(
                source_type=source_type_name,
                source_id=row["source_data_identifier"],
                target_type=target_type_name,
                target_id=row["target_data_identifier"],
                relation_type=relation_type,
            )
