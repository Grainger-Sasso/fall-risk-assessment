"""Audit agreement between the SQLite feature index and the payloads on disk.

Validating only the records that happen to load silently overstates coverage: an
index row whose payload was deleted disappears from every downstream count. This
module reports that drift explicitly so "are features generated for all my users?"
is answered against reality rather than against the index.

Detection is read-only. Nothing here deletes rows or directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.type_registry import IdentifierTypeRegistry
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier

FEATURE_PAYLOAD_FILENAME = "features.h5"


@dataclass
class IndexIntegrityReport:
    """Drift between indexed feature records, payloads on disk, and IMU coverage."""

    indexed_feature_count: int = 0
    loadable_feature_count: int = 0
    dangling_feature_ids: List[str] = field(default_factory=list)
    unreadable_feature_ids: Dict[str, str] = field(default_factory=dict)
    orphaned_payload_dirs: List[str] = field(default_factory=list)
    indexed_imu_count: int = 0
    imu_ids_without_features: List[str] = field(default_factory=list)
    participants_with_features: Set[str] = field(default_factory=set)

    @property
    def dangling_count(self) -> int:
        return len(self.dangling_feature_ids)

    @property
    def unreadable_count(self) -> int:
        return len(self.unreadable_feature_ids)

    @property
    def orphaned_count(self) -> int:
        return len(self.orphaned_payload_dirs)

    @property
    def is_clean(self) -> bool:
        return (
            self.dangling_count == 0
            and self.unreadable_count == 0
            and self.orphaned_count == 0
            and not self.imu_ids_without_features
        )

    @property
    def index_overstatement(self) -> int:
        """How many more records the index claims than can actually be read."""
        return self.indexed_feature_count - self.loadable_feature_count

    def summary_text(self) -> str:
        lines = [
            "Indexed feature records : {}".format(self.indexed_feature_count),
            "Loadable                : {}".format(self.loadable_feature_count),
            "Dangling (no payload)   : {}".format(self.dangling_count),
            "Unreadable payloads     : {}".format(self.unreadable_count),
            "Orphaned payload dirs   : {}".format(self.orphaned_count),
            "Indexed IMU records     : {}".format(self.indexed_imu_count),
            "IMU without features    : {}".format(len(self.imu_ids_without_features)),
            "Participants w/ features: {}".format(len(self.participants_with_features)),
        ]
        if self.index_overstatement > 0:
            lines.append(
                "WARNING: index overstates feature coverage by {} records "
                "({:.0f}%). Counts taken from the index alone are not trustworthy.".format(
                    self.index_overstatement,
                    100.0 * self.index_overstatement / max(self.loadable_feature_count, 1),
                )
            )
        return "\n".join(lines)


def _record_path(db_manager: DatabaseManager, feature_id: FeatureIdentifier) -> Optional[Path]:
    """Resolve the indexed payload directory, or None when the row has no path."""
    id_type = IdentifierTypeRegistry.get_type_name(type(feature_id))
    try:
        return db_manager.repository.get_record_path(id_type, feature_id.value)
    except Exception:
        return None


def audit_index_integrity(
    db_manager: DatabaseManager,
    feature_payload_roots: Optional[List[Path]] = None,
) -> IndexIntegrityReport:
    """Compare indexed feature records against payloads on disk.

    ``feature_payload_roots`` are directories to scan for orphaned payloads, i.e.
    feature directories present on disk with no corresponding index row. When
    omitted, orphan detection is skipped and only dangling rows are reported.
    """
    report = IndexIntegrityReport()

    feature_ids = db_manager.list_feature_ids()
    report.indexed_feature_count = len(feature_ids)

    indexed_paths: Set[Path] = set()
    imu_ids_with_features: Set[str] = set()

    for feature_id in feature_ids:
        path = _record_path(db_manager, feature_id)
        if path is not None:
            # The index stores resolved paths; roots may arrive relative or via symlinks.
            indexed_paths.add(path.resolve())
        if path is None or not path.exists():
            report.dangling_feature_ids.append(feature_id.value)
            continue
        try:
            record = db_manager.load_features(feature_id)
        except Exception as exc:
            report.unreadable_feature_ids[feature_id.value] = "{}: {}".format(
                type(exc).__name__, exc
            )
            continue
        report.loadable_feature_count += 1
        report.participants_with_features.add(record.feature_metadata.user_identifier.value)
        imu_ids_with_features.add(record.feature_metadata.imu_data_identifier.value)

    try:
        imu_ids: List[IMUDataIdentifier] = db_manager.list_imu_ids()
    except Exception:
        imu_ids = []
    report.indexed_imu_count = len(imu_ids)
    report.imu_ids_without_features = sorted(
        imu_id.value for imu_id in imu_ids if imu_id.value not in imu_ids_with_features
    )

    for root in feature_payload_roots or []:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for child in sorted(root_path.iterdir()):
            if child.is_dir() and child.resolve() not in indexed_paths:
                report.orphaned_payload_dirs.append(str(child))

    return report
