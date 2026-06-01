"""Build SQLite metadata indexes from assessment data folders."""

from pathlib import Path
from typing import Optional, Sequence, Union

from src.data_io.builders.model_builders.instrument_specifications.instrument_specification_builder import (
    IMUSpecificationBuilder,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataFileNames,
    IMUDataImporter,
)
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataFileNames,
    UserDataImporter,
)
from src.data_io.import_export.importers.features.feature_importer import (
    FeatureFileNames,
    FeatureImporter,
)
from src.data_io.read_write.readers.json.json_dict_file_reader import JSONDictFileReader
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.relation_types import (
    FEATURE_TO_IMU,
    IMU_TO_INSTRUMENT_SPEC,
    IMU_TO_USER,
)
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.database_manager.metadata.type_registry import IdentifierTypeRegistry
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier

IMU_DATA_FILE = f"{IMUDataFileNames.IMU_DATA.value}.h5"
USER_DATA_FILE = f"{UserDataFileNames.USER_DATA.value}.json"
CLINICAL_DEMO_FILE = f"{UserDataFileNames.CLININCAL_DEMOGRAPHIC_DATA.value}.json"
FEATURE_DATA_FILE = f"{FeatureFileNames.FEATURES.value}.h5"


def _validate_parent_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} parent directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} parent path must be a directory: {path}")


def _get_subdirs_with_required_files(
    parent_dir: Path, required_filenames: Sequence[str]
) -> list[Path]:
    subdirs = [
        d
        for d in parent_dir.iterdir()
        if d.is_dir() and all((d / filename).exists() for filename in required_filenames)
    ]
    if not subdirs:
        required = ", ".join(required_filenames)
        raise FileNotFoundError(
            f"No subdirectories with required files ({required}) found in {parent_dir}"
        )
    return sorted(subdirs)


def _normalize_instrument_spec_files(
    instrument_spec_inputs: Union[Path, Sequence[Path]]
) -> list[Path]:
    input_paths: list[Path]
    if isinstance(instrument_spec_inputs, Path):
        input_paths = [instrument_spec_inputs]
    else:
        input_paths = list(instrument_spec_inputs)

    if not input_paths:
        raise ValueError("At least one instrument specification input must be provided")

    spec_files: list[Path] = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Instrument spec input not found: {input_path}")
        if input_path.is_file():
            if input_path.suffix != ".json":
                raise ValueError(f"Instrument spec file must be JSON: {input_path}")
            spec_files.append(input_path.resolve())
            continue

        for candidate in input_path.rglob("*.json"):
            spec_files.append(candidate.resolve())

    if not spec_files:
        raise FileNotFoundError("No instrument specification JSON files found")
    return sorted(set(spec_files))


def _upsert_record(
    repository: MetadataRepository, identifier_type: type, identifier_value: str, path: Path
) -> None:
    id_type_name = IdentifierTypeRegistry.get_type_name(identifier_type)
    repository.upsert_record(
        id_type=id_type_name,
        identifier=identifier_value,
        path=path.resolve(),
    )


def _add_relation(
    repository: MetadataRepository,
    source_type: type,
    source_id: str,
    target_type: type,
    target_id: str,
    relation_type: str,
) -> None:
    repository.add_relation(
        source_type=IdentifierTypeRegistry.get_type_name(source_type),
        source_id=source_id,
        target_type=IdentifierTypeRegistry.get_type_name(target_type),
        target_id=target_id,
        relation_type=relation_type,
    )


def build_sql_index(
    imu_parent_dir: Path,
    user_parent_dir: Path,
    instrument_spec_inputs: Union[Path, Sequence[Path]],
    sqlite_db_path: Path,
    feature_parent_dir: Optional[Path] = None,
) -> Path:
    """Build SQLite records/relations indexes from filesystem data payloads."""
    _validate_parent_dir(imu_parent_dir, "IMU")
    _validate_parent_dir(user_parent_dir, "User")
    if feature_parent_dir is not None:
        _validate_parent_dir(feature_parent_dir, "Feature")

    sqlite_store = SQLiteStore(sqlite_db_path.resolve())
    repository = MetadataRepository(sqlite_store)

    imu_importer = IMUDataImporter()
    user_importer = UserDataImporter()
    feature_importer = FeatureImporter()
    json_reader = JSONDictFileReader()
    spec_builder = IMUSpecificationBuilder()

    imu_dirs = _get_subdirs_with_required_files(imu_parent_dir, [IMU_DATA_FILE])
    user_dirs = _get_subdirs_with_required_files(
        user_parent_dir, [USER_DATA_FILE, CLINICAL_DEMO_FILE]
    )
    feature_dirs = []
    if feature_parent_dir is not None:
        feature_dirs = _get_subdirs_with_required_files(feature_parent_dir, [FEATURE_DATA_FILE])
    spec_files = _normalize_instrument_spec_files(instrument_spec_inputs)

    spec_ids: list[InstrumentSpecificationIdentifier] = []
    spec_name_to_id: dict[str, InstrumentSpecificationIdentifier] = {}
    for spec_file in spec_files:
        spec_json = json_reader.read(spec_file)
        specification = spec_builder.build(spec_json)
        spec_id = specification.specification_id
        spec_ids.append(spec_id)
        spec_name_to_id[specification.imu_name.strip().lower()] = spec_id
        _upsert_record(repository, InstrumentSpecificationIdentifier, spec_id.value, spec_file.parent)

    imu_to_instrument_name: dict[str, str] = {}
    for imu_dir in imu_dirs:
        imu_data = imu_importer.import_data(imu_dir)
        imu_id = imu_data.get_data_id()
        user_id = imu_data.get_associated_data_id()
        instrument_name = imu_data.metadata.instrument_identifier.name

        _upsert_record(repository, IMUDataIdentifier, imu_id.value, imu_dir)
        _add_relation(
            repository,
            IMUDataIdentifier,
            imu_id.value,
            UserIdentifier,
            user_id.value,
            IMU_TO_USER,
        )
        imu_to_instrument_name[imu_id.value] = instrument_name

    for user_dir in user_dirs:
        user_data = user_importer.import_data(user_dir)
        user_id = user_data.get_data_id()
        _upsert_record(repository, UserIdentifier, user_id.value, user_dir)

    for feature_dir in feature_dirs:
        feature_data = feature_importer.import_data(feature_dir)
        feature_id = feature_data.get_data_id()
        imu_id = feature_data.get_associated_data_id()
        _upsert_record(repository, FeatureIdentifier, feature_id.value, feature_dir)
        _add_relation(
            repository,
            FeatureIdentifier,
            feature_id.value,
            IMUDataIdentifier,
            imu_id.value,
            FEATURE_TO_IMU,
        )

    unresolved_imu_ids: list[str] = []
    if len(spec_ids) == 1:
        singleton_spec = spec_ids[0]
        for imu_id_value in imu_to_instrument_name:
            _add_relation(
                repository,
                IMUDataIdentifier,
                imu_id_value,
                InstrumentSpecificationIdentifier,
                singleton_spec.value,
                IMU_TO_INSTRUMENT_SPEC,
            )
    else:
        for imu_id_value, instrument_name in imu_to_instrument_name.items():
            lookup_key = instrument_name.strip().lower()
            if lookup_key in spec_name_to_id:
                _add_relation(
                    repository,
                    IMUDataIdentifier,
                    imu_id_value,
                    InstrumentSpecificationIdentifier,
                    spec_name_to_id[lookup_key].value,
                    IMU_TO_INSTRUMENT_SPEC,
                )
            else:
                unresolved_imu_ids.append(imu_id_value)

    if unresolved_imu_ids:
        unresolved_text = ", ".join(sorted(unresolved_imu_ids))
        print(
            "Warning: no instrument specification relation added for IMU IDs: "
            f"{unresolved_text}"
        )

    print(f"SQLite metadata index created at: {sqlite_db_path.resolve()}")
    return sqlite_db_path.resolve()


def main() -> None:
    build_sql_index(
        imu_parent_dir=Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/assessment_database/imu_data/lab_walks"
        ),
        user_parent_dir=Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/assessment_database/user_data/lab_walks"
        ),
        instrument_spec_inputs=Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/assessment_database/instrument_specs/lab_walks/instrument_spec.json"
        ),
        sqlite_db_path=Path(
            "/Users/graingersasso/Desktop/fafra/fafra_data/assessment_database/sql_db_indexes/lab_walks/index.db"
        ),
    )


if __name__ == "__main__":
    main()
