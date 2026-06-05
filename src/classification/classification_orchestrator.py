"""CLI orchestration for classification model evaluation on the new sample basis.

Builds stride/epoch classification datasets from a SQLite-backed
DatabaseManager, runs the late-fusion evaluator across model families and fusion
strategies, writes a JSON evaluation artifact, and optionally renders a PDF
comparison report.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.classification.data.classification_dataset_builder import (
    ClassificationDatasetBuilder,
)
from src.classification.evaluation.evaluation_artifact import EvaluationArtifact
from src.classification.evaluation.fusion_evaluator import FusionEvaluator
from src.classification.fusion.fusion_strategies import available_fusion_names
from src.classification.models.model_factory import available_model_names
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

ARTIFACT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def build_database_manager(sqlite_db_path: Path) -> DatabaseManager:
    """Wire a read-oriented DatabaseManager over an existing index DB."""
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


def _artifact_path(output_dir: Path, fast_mode: bool, generated_at: datetime) -> Path:
    timestamp = generated_at.strftime(ARTIFACT_TIMESTAMP_FORMAT)
    mode = "fast" if fast_mode else "full"
    return output_dir / f"classification_evaluation_{mode}_{timestamp}.json"


def run_classification_evaluation(
    sqlite_db_path: Path,
    output_dir: Path,
    model_names: Optional[List[str]] = None,
    fusion_names: Optional[List[str]] = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    fast_mode: bool = False,
    generate_report: bool = False,
) -> EvaluationArtifact:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[classification] Building datasets from {sqlite_db_path}")
    db_manager = build_database_manager(sqlite_db_path)
    dataset = ClassificationDatasetBuilder(db_manager=db_manager).build()
    print(
        f"[classification] stride samples={dataset.stride.n_samples}, "
        f"epoch samples={dataset.epoch.n_samples}, "
        f"common participants={len(dataset.common_participant_ids())}"
    )

    evaluator = FusionEvaluator(
        dataset=dataset,
        model_names=model_names or available_model_names(),
        fusion_names=fusion_names or available_fusion_names(),
        n_splits=n_splits,
        n_repeats=n_repeats,
        inner_splits=inner_splits,
        random_state=random_state,
        fast_mode=fast_mode,
    )
    artifact = evaluator.evaluate()

    generated_at = datetime.now()
    artifact_path = _artifact_path(output_dir, fast_mode, generated_at)
    artifact.to_json(artifact_path)
    print(f"[classification] Evaluation artifact written to: {artifact_path}")

    if artifact.ranking:
        best = artifact.ranking[0]
        print(
            "[classification] Top configuration: "
            f"{best['model_name']} + {best['fusion']} "
            f"(score={best['score']:.4f}, roc_auc={best['roc_auc_mean']})"
        )

    if generate_report:
        from src.data_visualization.reporting.classification_report_generator import (
            generate_classification_report,
        )

        report_pdf = generate_classification_report(
            artifact=artifact,
            output_pdf=output_dir / "classification_report.pdf",
        )
        print(f"[classification] Report PDF written to: {report_pdf}")

    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate late-fusion fall-risk classifiers on the sample basis."
    )
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model family names. Empty means all available.",
    )
    parser.add_argument(
        "--fusions",
        type=str,
        default="",
        help="Comma-separated fusion strategy names. Empty means all available.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--report", action="store_true", help="Also render a PDF report.")
    return parser.parse_args()


def _split_csv(value: str) -> Optional[List[str]]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main() -> None:
    args = parse_args()
    run_classification_evaluation(
        sqlite_db_path=args.sqlite_db_path,
        output_dir=args.output_dir,
        model_names=_split_csv(args.models),
        fusion_names=_split_csv(args.fusions),
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
        fast_mode=args.fast_mode,
        generate_report=args.report,
    )


if __name__ == "__main__":
    main()
