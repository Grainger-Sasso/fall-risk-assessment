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
from src.data_model.data.imu.epoch_imu_data import EpochIMUData
from src.data_model.data.imu.imu_data import IMUData
from src.database_manager.mapping.mapping import Mapping
from src.database_manager.registry.registry import Registry
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)

IMU_DATA_FILENAME = IMUDataFileNames.IMU_DATA.value
IMU_DATA_SUFFIX = ".h5"
IMU_DATA_FILE = IMU_DATA_FILENAME + IMU_DATA_SUFFIX

# Hard-coded paths - edit these to change inputs/outputs
PARENT_DIR = Path(
    "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/imu_data"
)
REGISTRIES_OUTPUT = Path(
    "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/registries"
)
MAPPINGS_OUTPUT = Path(
    "/Users/graingersasso/Desktop/fafra/fafra_data/converted_data/ltmm_2026_02_21/mappings"
)


def read_imu_data_from_parent(
    parent_dir: Path,
    registries_output: Path = REGISTRIES_OUTPUT,
    mappings_output: Path = MAPPINGS_OUTPUT,
) -> None:
    """
    Read imu_data.h5 files from subdirectories, build mapping and registry,
    and export both to the specified output directories. File data is not
    retained in memory.

    Args:
        parent_dir: Parent directory containing subdirectories, each with imu_data.h5
        registries_output: Directory to write registry.csv (IMU data ID -> path)
        mappings_output: Directory to write mapping.csv (user ID -> IMU data ID)

    Raises:
        FileNotFoundError: If parent_dir does not exist or no subdirs with imu_data.h5 found.
    """
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")

    subdirs_with_imu = [
        d for d in parent_dir.iterdir() if d.is_dir() and (d / IMU_DATA_FILE).exists()
    ]

    if not subdirs_with_imu:
        raise FileNotFoundError(
            f"No subdirectories with {IMU_DATA_FILE} found in {parent_dir}"
        )

    importer = IMUDataImporter()

    # Mapping: source (user ID) -> target (IMU data ID)
    user_to_imu_map: dict[str, str] = {}
    # Registry: IMU data ID -> directory path (where imu_data.h5 lives)
    imu_id_to_path_registry: dict[str, Path] = {}
    # IMU ID to data value map
    imu_id_to_data_value_map: dict[str, float] = {}

    for subdir in sorted(subdirs_with_imu):
        imu_data: IMUData = importer.import_data(subdir)
        epoch_data: EpochIMUData = imu_data.data[0]
        imu_id = imu_data.get_data_id()
        user_id = imu_data.get_associated_data_id()
        print(f"Fetched file for IMU data: {imu_id.value}")

        user_to_imu_map[user_id.value] = imu_id.value
        imu_id_to_path_registry[imu_id.value] = subdir.resolve()
        imu_id_to_data_value_map[imu_id.value] = (
            epoch_data.data[0]
            .get_data_by_sensor_axis(sensor_axis=SensorCoordinateSystem.X)
            .data[0]
        )

    # Build Registry (IMU data ID -> directory path)
    registry = Registry(
        registry=imu_id_to_path_registry,
        id_type=IMUDataIdentifier,
        subdir_path=registries_output,
    )

    # Build Mapping (user ID -> IMU data ID)
    mapping = Mapping(
        map=user_to_imu_map,
        source_id_type=UserIdentifier,
        target_id_type=IMUDataIdentifier,
        subdir_path=mappings_output,
    )

    # Export both to disk
    registries_output.mkdir(parents=True, exist_ok=True)
    mappings_output.mkdir(parents=True, exist_ok=True)

    registry_exporter = RegistryExporter()
    mapping_exporter = MappingExporter()

    registry_exporter.export_data(registries_output, registry)
    mapping_exporter.export_data(mappings_output, mapping)

    print(f"Registry written to {registries_output / 'registry.csv'}")
    print(f"Mapping written to {mappings_output / 'mapping.csv'}")
    print(f"Processed {len(imu_id_to_path_registry)} IMU data files")


def main() -> None:
    read_imu_data_from_parent(
        PARENT_DIR,
        registries_output=REGISTRIES_OUTPUT,
        mappings_output=MAPPINGS_OUTPUT,
    )


if __name__ == "__main__":
    main()
