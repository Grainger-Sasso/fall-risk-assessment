"""
Read IMU data files from a parent directory and build registry and mapping.

Input: A single parent directory path. The parent contains subdirectories,
       each with a single file named "imu_data.h5".

Reads each file to extract IDs, builds a mapping (user ID -> IMU data ID) and
registry (IMU data ID -> directory path), then exports both. Does not retain
file data in memory.

Usage:
    python -m src.util.dataset_generator.read_imu_data

Edit the hard-coded paths below to change inputs/outputs.
"""

from pathlib import Path

from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.import_export.exporters.mapping.mapping_exporter import (
    MappingExporter,
)
from src.data_io.import_export.exporters.registry.registry_exporter import (
    RegistryExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataFileNames,
    IMUDataImporter,
)
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataFileNames,
)
from src.data_io.model_fields.dataset.dataset_fields import DatasetFields
from src.data_io.read_write.writers.csv.csv_file_writer import CSVFileWriter
from src.data_model.data.imu.imu_data import IMUData
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier

IMU_DATA_FILENAME = IMUDataFileNames.IMU_DATA.value
IMU_DATA_SUFFIX = ".h5"
IMU_DATA_FILE = IMU_DATA_FILENAME + IMU_DATA_SUFFIX

USER_DATA_FILENAME = UserDataFileNames.USER_DATA.value
USER_DATA_SUFFIX = ".json"
USER_DATA_FILE = USER_DATA_FILENAME + USER_DATA_SUFFIX

IMU_SUBDIR = "imu_data"
USER_SUBDIR = "user_data"
REGISTRIES_SUBDIR = "registries"
MAPPINGS_SUBDIR = "mappings"
DATASET_SUBDIR = "dataset"

IMU_REG_SUBDIR = "imu_data"
USER_REG_SUBDIR = "user_data"
RAW_REG_SUBDIR = "raw_feature"
AGG_REG_SUBDIR = "agg_feature"

IMU_MAP_SUBDIR = "imu_to_user_mapping"
RAW_MAP_SUBDIR = "raw_feat_to_imu_mapping"
AGG_MAP_SUBDIR = "agg_feat_to_raw_feat_mapping"


def _get_subdirs_with_required_file(
    parent_dir: Path, required_filename: str
) -> list[Path]:
    """
    Validate parent directory and collect subdirectories containing a required file.
    """
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")

    subdirs = [
        d
        for d in parent_dir.iterdir()
        if d.is_dir() and (d / required_filename).exists()
    ]
    if not subdirs:
        raise FileNotFoundError(
            f"No subdirectories with {required_filename} found in {parent_dir}"
        )
    return subdirs


def _write_dataset_csv(
    imu_to_user_map: dict[str, str],
    dataset_output: Path,
) -> Path:
    """Write dataset.csv mapping user IDs to IMU IDs."""
    if not dataset_output.exists():
        raise FileNotFoundError(f"Required output directory not found: {dataset_output}")
    if not dataset_output.is_dir():
        raise NotADirectoryError(f"Expected directory path: {dataset_output}")
    imu_ids = imu_to_user_map.keys()
    dataset_csv = CSVFile(
        fieldnames=[
            DatasetFields.USER_DATA_IDENTIFIER.value,
            DatasetFields.IMU_DATA_IDENTIFIER.value,
        ],
        data={
            DatasetFields.USER_DATA_IDENTIFIER.value: [
                imu_to_user_map[imu_id] for imu_id in imu_ids
            ],
            DatasetFields.IMU_DATA_IDENTIFIER.value: imu_ids,
        },
    )
    dataset_writer = CSVFileWriter()
    dataset_path = dataset_output / "dataset.csv"
    write_success, error_msg = dataset_writer.write(dataset_path, dataset_csv)
    if not write_success:
        raise RuntimeError(f"Failed writing dataset CSV: {error_msg}")
    return dataset_path


def _require_existing_subdir(root_dir: Path, subdir_name: str) -> Path:
    """Resolve required direct child directory under root and validate it exists."""
    path = root_dir / subdir_name
    if not path.exists():
        raise FileNotFoundError(f"Required subdir not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory path: {path}")
    return path


def _get_or_create_subdir(root_dir: Path, subdir_name: str) -> Path:
    """Resolve direct child directory under root, creating it if needed."""
    path = root_dir / subdir_name
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory path: {path}")
    return path


def _require_empty_dir(path: Path, label: str) -> None:
    """Ensure directory exists and is empty before writing outputs."""
    if any(path.iterdir()):
        raise ValueError(f"Expected empty {label} directory, but found contents: {path}")


def build_dataset(root_dir: Path) -> None:
    """
    Read imu_data.h5 files from subdirectories, build mapping and registry,
    and export both to the specified output directories. File data is not
    retained in memory.

    Args:
        root_dir: Converted-data root directory containing required subdirs.
    """
    print("Dataset generation started")
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Expected directory path: {root_dir}")

    imu_parent_dir = _require_existing_subdir(root_dir, IMU_SUBDIR)
    user_parent_dir = _require_existing_subdir(root_dir, USER_SUBDIR)

    # Create output directory structure if missing.
    registries_root = _get_or_create_subdir(root_dir, REGISTRIES_SUBDIR)
    mappings_root = _get_or_create_subdir(root_dir, MAPPINGS_SUBDIR)
    dataset_output_path = _get_or_create_subdir(root_dir, DATASET_SUBDIR)

    imu_reg_ouput_path = _get_or_create_subdir(registries_root, IMU_REG_SUBDIR)
    user_reg_ouput_path = _get_or_create_subdir(registries_root, USER_REG_SUBDIR)
    raw_reg_ouput_path = _get_or_create_subdir(registries_root, RAW_REG_SUBDIR)
    agg_reg_ouput_path = _get_or_create_subdir(registries_root, AGG_REG_SUBDIR)

    imu_map_output_path = _get_or_create_subdir(mappings_root, IMU_MAP_SUBDIR)
    raw_map_output_path = _get_or_create_subdir(mappings_root, RAW_MAP_SUBDIR)
    agg_map_output_path = _get_or_create_subdir(mappings_root, AGG_MAP_SUBDIR)

    # Output directories must be empty before generation.
    _require_empty_dir(imu_reg_ouput_path, "registry")
    _require_empty_dir(user_reg_ouput_path, "registry")
    _require_empty_dir(raw_reg_ouput_path, "registry")
    _require_empty_dir(agg_reg_ouput_path, "registry")
    _require_empty_dir(imu_map_output_path, "mapping")
    _require_empty_dir(raw_map_output_path, "mapping")
    _require_empty_dir(agg_map_output_path, "mapping")
    _require_empty_dir(dataset_output_path, "dataset")

    # Must be non-empty with valid data subdirectories.
    imu_subdirs = _get_subdirs_with_required_file(imu_parent_dir, IMU_DATA_FILE)
    user_subdirs = _get_subdirs_with_required_file(user_parent_dir, USER_DATA_FILE)

    imu_importer = IMUDataImporter()

    # Mapping: source (user ID) -> target (IMU data ID)
    imu_to_user_map: dict[str, str] = {}
    # Registry: IMU data ID -> directory path (where imu_data.h5 lives)
    imu_id_to_path_registry: dict[str, Path] = {}

    for imu_subdir in sorted(imu_subdirs):
        imu_data: IMUData = imu_importer.import_data(imu_subdir)
        imu_id = imu_data.get_data_id()
        user_id = imu_data.get_associated_data_id()
        print(f"Fetched file for IMU data: {imu_id.value}")

        imu_to_user_map[imu_id.value] = user_id.value
        imu_id_to_path_registry[imu_id.value] = imu_subdir.resolve()

    user_id_to_path_registry: dict[str, Path] = {}
    for user_subdir in sorted(user_subdirs):
        # Get user ID from subdir name in format: user_USERID
        prefix = "user_"
        subdir_name = user_subdir.name
        if not subdir_name.startswith(prefix):
            raise ValueError(
                f"Invalid user subdir name '{subdir_name}'. Expected format: user_USERID"
            )
        user_id = subdir_name[len(prefix) :]
        if not user_id:
            raise ValueError(
                f"Invalid user subdir name '{subdir_name}'. USERID cannot be empty."
            )
        user_id_to_path_registry[user_id] = user_subdir.resolve()

    # Build Dataset CSV: output mapping between user IDs and IMU IDs
    dataset_path = _write_dataset_csv(imu_to_user_map, dataset_output_path)

    # Build IMU Registry (IMU data ID -> directory path)
    imu_registry = Registry(
        registry=imu_id_to_path_registry,
        id_type=IMUDataIdentifier,
        subdir_path=imu_reg_ouput_path,
    )
    # Build USER Registry (USER ID -> directory path)
    user_registry = Registry(
        registry=user_id_to_path_registry,
        id_type=UserIdentifier,
        subdir_path=user_reg_ouput_path,
    )
    # Build Raw Feature Registry - WILL BE BLANK (Raw Feat ID -> directory path)
    raw_feat_registry = Registry(
        registry={}, id_type=RawFeatureIdentifier, subdir_path=raw_reg_ouput_path
    )
    # Build Agg Feature Registry - WILL BE BLANK (Agg Feat ID -> directory path)
    agg_feat_registry = Registry(
        registry={}, id_type=AggregateFeatureIdentifier, subdir_path=agg_reg_ouput_path
    )
    # Export registries
    registry_exporter = RegistryExporter()
    registry_exporter.export_data(imu_reg_ouput_path, imu_registry)
    registry_exporter.export_data(user_reg_ouput_path, user_registry)
    registry_exporter.export_data(raw_reg_ouput_path, raw_feat_registry)
    registry_exporter.export_data(agg_reg_ouput_path, agg_feat_registry)

    # Build Mapping (IMU data ID -> user ID)
    imu_to_user_mapping = Mapping(
        map=imu_to_user_map,
        source_id_type=IMUDataIdentifier,
        target_id_type=UserIdentifier,
        subdir_path=imu_map_output_path,
    )
    # Build Mapping (raw feat ID -> imu data ID)
    raw_feat_to_imu_mapping = Mapping(
        map={},
        source_id_type=RawFeatureIdentifier,
        target_id_type=IMUDataIdentifier,
        subdir_path=raw_map_output_path,
    )
    # Build Mapping (agg feat ID -> raw feat ID)
    agg_feat_to_raw_feat_mapping = Mapping(
        map={},
        source_id_type=AggregateFeatureIdentifier,
        target_id_type=RawFeatureIdentifier,
        subdir_path=agg_map_output_path,
    )

    # Export Mappings
    mapping_exporter = MappingExporter()
    mapping_exporter.export_data(imu_map_output_path, imu_to_user_mapping)
    mapping_exporter.export_data(raw_map_output_path, raw_feat_to_imu_mapping)
    mapping_exporter.export_data(agg_map_output_path, agg_feat_to_raw_feat_mapping)

    print(f"Dataset written to {dataset_path}")
    print("Dataset generation completed")


def main() -> None:
    root_dir = Path(
        "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_lab_walks_2026_05_09"
    )

    build_dataset(root_dir)


if __name__ == "__main__":
    main()
