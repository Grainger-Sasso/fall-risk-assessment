"""Locate and load real participant records from the external assessment database."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from src.data_io.import_export.importers.data.imu.imu_data_importer import IMUDataImporter
from src.data_io.import_export.importers.data.user.user_data_importer import UserDataImporter
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.database_manager.database_manager import DatabaseManager
from src.gait_features.gait_feature_dataset_orchestrator import build_database_manager_from_sqlite
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier

CATALOG_PATH = Path(__file__).with_name("assessment_fixture_catalog.json")
ENV_ROOT = "FRA_ASSESSMENT_DATABASE_ROOT"
ENV_RUN_LTMM = "RUN_LTMM_INTEGRATION_TESTS"
DATASETS: Tuple[str, ...] = ("ltmm", "lab_walks")


@dataclass(frozen=True)
class ParticipantFixture:
    dataset: str
    participant_id: str
    imu_id: str
    tier: str
    expects_walking_bouts: bool = False


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def assessment_database_root() -> Path:
    override = os.environ.get(ENV_ROOT, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(load_catalog()["assessment_database_root"]).expanduser().resolve()


def _path_is_readable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb"):
            return True
    except OSError:
        return False


def _dataset_config(dataset: str) -> dict:
    catalog = load_catalog()
    if dataset not in catalog:
        raise KeyError(f"Unknown fixture dataset '{dataset}'")
    return catalog[dataset]


def participant_fixtures(dataset: str, tier: Optional[str] = None) -> List[ParticipantFixture]:
    config = _dataset_config(dataset)
    fixtures: List[ParticipantFixture] = []
    for participant_id, entry in config["participants"].items():
        if tier is not None and entry.get("tier") != tier:
            continue
        fixtures.append(
            ParticipantFixture(
                dataset=dataset,
                participant_id=participant_id,
                imu_id=entry["imu_id"],
                tier=entry.get("tier", "primary"),
                expects_walking_bouts=bool(entry.get("expects_walking_bouts", False)),
            )
        )
    return fixtures


def index_db_path(dataset: str) -> Path:
    return assessment_database_root() / _dataset_config(dataset)["index_db"]


def instrument_spec_dir(dataset: str) -> Path:
    return assessment_database_root() / _dataset_config(dataset)["instrument_spec_dir"]


def imu_record_dir(dataset: str, imu_id: str) -> Path:
    subdir = "ltmm" if dataset == "ltmm" else "lab_walks"
    return assessment_database_root() / "imu_data" / subdir / f"imu_data_{imu_id}"


def user_record_dir(dataset: str, participant_id: str) -> Path:
    subdir = "ltmm" if dataset == "ltmm" else "lab_walks"
    return assessment_database_root() / "user_data" / subdir / f"user_{participant_id}"


def assessment_database_errors(datasets: Tuple[str, ...] = DATASETS) -> List[str]:
    """Return human-readable reasons why the assessment-database fixtures are unavailable."""
    root = assessment_database_root()
    if not root.is_dir():
        return [
            f"Assessment database root is not accessible: {root} "
            f"(mount the backup volume or set {ENV_ROOT})"
        ]

    errors: List[str] = []
    for dataset in datasets:
        config = _dataset_config(dataset)
        index_path = root / config["index_db"]
        if not _path_is_readable_file(index_path):
            errors.append(f"Missing or unreadable SQLite index: {index_path}")

        spec_file = root / config["instrument_spec_dir"] / "instrument_spec.json"
        if not _path_is_readable_file(spec_file):
            errors.append(f"Missing or unreadable instrument spec: {spec_file}")

        primary = participant_fixtures(dataset, tier="primary")
        if not primary:
            errors.append(f"No primary participants configured for dataset '{dataset}'")
            continue
        fixture = primary[0]
        imu_h5 = imu_record_dir(dataset, fixture.imu_id) / "imu_data.h5"
        if not _path_is_readable_file(imu_h5):
            errors.append(f"Missing or unreadable IMU fixture: {imu_h5}")

        user_dir = user_record_dir(dataset, fixture.participant_id)
        for filename in ("user_data.json", "clinical_demographic_data.json"):
            user_file = user_dir / filename
            if not _path_is_readable_file(user_file):
                errors.append(f"Missing or unreadable user fixture: {user_file}")

    return errors


def ltmm_integration_requested() -> bool:
    return os.environ.get(ENV_RUN_LTMM, "").strip().lower() in {"1", "true", "yes"}


def load_imu(dataset: str, imu_id: str) -> IMUData:
    return IMUDataImporter().import_data(imu_record_dir(dataset, imu_id))


def load_user(dataset: str, participant_id: str) -> UserData:
    return UserDataImporter().import_data(user_record_dir(dataset, participant_id))


def load_instrument_spec(dataset: str) -> IMUSpecifications:
    return InstrumentSpecificationImporter().import_data(instrument_spec_dir(dataset))


def load_participant_bundle(
    fixture: ParticipantFixture,
) -> Tuple[IMUData, UserData, IMUSpecifications]:
    return (
        load_imu(fixture.dataset, fixture.imu_id),
        load_user(fixture.dataset, fixture.participant_id),
        load_instrument_spec(fixture.dataset),
    )


def build_database_manager(
    dataset: str,
    *,
    feature_output_dir: Optional[Path] = None,
    sqlite_copy_dir: Optional[Path] = None,
) -> DatabaseManager:
    source_index = index_db_path(dataset)
    if sqlite_copy_dir is not None:
        sqlite_copy_dir.mkdir(parents=True, exist_ok=True)
        target_index = sqlite_copy_dir / "index.db"
        shutil.copy2(source_index, target_index)
        sqlite_db_path = target_index
        default_output = sqlite_copy_dir
    else:
        sqlite_db_path = source_index
        default_output = source_index.parent

    return build_database_manager_from_sqlite(
        sqlite_db_path=sqlite_db_path,
        feature_output_dir=feature_output_dir or default_output,
    )


@dataclass
class TempDatabaseContext:
    root: Path
    db_manager: DatabaseManager

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


@contextmanager
def temp_database_manager(dataset: str) -> Iterator[TempDatabaseContext]:
    """Isolated copy of a dataset index with no feature rows; payloads write to a temp dir."""
    root = Path(tempfile.mkdtemp(prefix=f"fra_test_{dataset}_"))
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    db_manager = build_database_manager(
        dataset,
        feature_output_dir=feature_dir,
        sqlite_copy_dir=root,
    )
    # Copied production indexes may reference pre-existing feature payloads.
    db_manager.cleanup_all_features(delete_payloads=False)
    context = TempDatabaseContext(root=root, db_manager=db_manager)
    try:
        yield context
    finally:
        context.cleanup()


def imu_identifier(imu_id: str) -> IMUDataIdentifier:
    return IMUDataIdentifier(imu_id)


def user_identifier(participant_id: str) -> UserIdentifier:
    return UserIdentifier(participant_id)
