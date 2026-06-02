import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.exporters.feature.feature_file_exporter import (
    FeatureFileExporter,
)
from src.data_io.import_export.exporters.instrument_specifications.instrument_specification_file_exporter import (
    InstrumentSpecificationFileExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import IMUDataImporter
from src.data_io.import_export.importers.data.user.user_data_importer import UserDataImporter
from src.data_io.import_export.importers.features.feature_importer import FeatureImporter
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.gait_features.gait_feature_module import (
    GaitFeatureExtractor,
    RecordFeatureGenerationBuilder,
)
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class FeatureGenerationRunSummary:
    generated_feature_ids: List[FeatureIdentifier] = field(default_factory=list)
    failed_imu_ids: List[IMUDataIdentifier] = field(default_factory=list)
    errors_by_imu_id: Dict[str, str] = field(default_factory=dict)


class GaitFeatureDatasetOrchestrator:
    """
    Orchestrates feature extraction for IMU datasets indexed in DatabaseManager.

    Flow per IMU:
      load IMU/user/spec -> SKDH extraction -> RecordFeatures build -> save_features
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        treadmill_profile: bool = False,
        extractor: Optional[GaitFeatureExtractor] = None,
        record_feature_builder: Optional[RecordFeatureGenerationBuilder] = None,
    ):
        self.db_manager = db_manager
        self.treadmill_profile = treadmill_profile
        self.extractor = extractor or GaitFeatureExtractor(
            treadmill_profile=treadmill_profile
        )
        self.record_feature_builder = record_feature_builder or RecordFeatureGenerationBuilder()

    def generate_for_all_imu(
        self, continue_on_error: bool = False
    ) -> FeatureGenerationRunSummary:
        return self.generate_for_imu_ids(
            imu_ids=self.db_manager.list_imu_ids(),
            continue_on_error=continue_on_error,
        )

    def generate_for_imu_ids(
        self,
        imu_ids: List[IMUDataIdentifier],
        continue_on_error: bool = False,
    ) -> FeatureGenerationRunSummary:
        summary = FeatureGenerationRunSummary()
        for imu_id in imu_ids:
            try:
                feature_id = self._generate_for_single_imu(imu_id)
                summary.generated_feature_ids.append(feature_id)
            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(
                        f"Feature generation failed for IMU '{imu_id.value}': {exc}"
                    ) from exc
                summary.failed_imu_ids.append(imu_id)
                summary.errors_by_imu_id[imu_id.value] = str(exc)
        return summary

    def _generate_for_single_imu(self, imu_id: IMUDataIdentifier) -> FeatureIdentifier:
        imu_data = self.db_manager.load_imu(imu_id)
        user_id = self.db_manager.get_user_for_imu(imu_id)
        if user_id is None:
            raise ValueError(f"No user mapping found for IMU '{imu_id.value}'.")
        user_data = self.db_manager.load_user(user_id)

        spec = None
        spec_id = self.db_manager.get_instrument_spec_for_imu(imu_id)
        if spec_id is not None:
            spec = self.db_manager.load_instrument_spec(spec_id)

        gait_results = self.extractor.extract_gait_features(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=spec,
        )
        record_features = self.record_feature_builder.build(
            gait_results=gait_results,
            imu_data=imu_data,
            user_data=user_data,
            treadmill_profile=self.treadmill_profile,
        )
        self.db_manager.save_features(record_features)
        return record_features.feature_metadata.feature_identifier


def build_database_manager_from_sqlite(sqlite_db_path: Path) -> DatabaseManager:
    """
    Build a DatabaseManager against an existing SQLite index for gait generation jobs.
    """
    sqlite_db_path = sqlite_db_path.resolve()
    scratch_dir = sqlite_db_path.parent
    repository = MetadataRepository(SQLiteStore(sqlite_db_path))
    io_router = DomainIORouter(
        imu_importer=IMUDataImporter(),
        user_importer=UserDataImporter(),
        feature_importer=FeatureImporter(),
        instrument_spec_importer=InstrumentSpecificationImporter(),
        imu_exporter=IMUDataFileExporter(),
        feature_exporter=FeatureFileExporter(),
        instrument_spec_exporter=InstrumentSpecificationFileExporter(),
        imu_output_dir=scratch_dir,
        feature_output_dir=scratch_dir,
        instrument_spec_output_dir=scratch_dir,
    )
    return DatabaseManager(metadata_repository=repository, io_router=io_router)


def _parse_imu_ids(raw_value: str) -> List[IMUDataIdentifier]:
    entries = [item.strip() for item in raw_value.split(",") if item.strip()]
    return [IMUDataIdentifier(value) for value in entries]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gait RecordFeatures from IMU/User data indexed in SQLite."
    )
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument(
        "--imu-ids",
        type=str,
        default="",
        help="Optional comma-separated IMU IDs to process. Empty means all indexed IMUs.",
    )
    parser.add_argument(
        "--treadmill-profile",
        action="store_true",
        help="Use treadmill SKDH profile for extraction.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining IMUs when a record fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_manager = build_database_manager_from_sqlite(args.sqlite_db_path)
    orchestrator = GaitFeatureDatasetOrchestrator(
        db_manager=db_manager,
        treadmill_profile=args.treadmill_profile,
    )

    if args.imu_ids.strip():
        imu_ids = _parse_imu_ids(args.imu_ids)
        summary = orchestrator.generate_for_imu_ids(
            imu_ids=imu_ids,
            continue_on_error=args.continue_on_error,
        )
    else:
        summary = orchestrator.generate_for_all_imu(
            continue_on_error=args.continue_on_error
        )

    print(f"Generated features: {len(summary.generated_feature_ids)}")
    if summary.generated_feature_ids:
        print("Generated feature IDs:")
        for feature_id in summary.generated_feature_ids:
            print(f"- {feature_id.value}")
    print(f"Failed IMUs: {len(summary.failed_imu_ids)}")
    if summary.failed_imu_ids:
        print("Failures:")
        for imu_id in summary.failed_imu_ids:
            print(f"- {imu_id.value}: {summary.errors_by_imu_id.get(imu_id.value, '')}")


if __name__ == "__main__":
    main()
