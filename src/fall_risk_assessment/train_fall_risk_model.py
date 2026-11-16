"""CLI: train and export a fall-risk model bundle from a feature database.

Example:
    python -m src.fall_risk_assessment.train_fall_risk_model \\
        --sqlite-db-path /data/sql_db_indexes/ltmm/index.db \\
        --output-dir /data/models/ltmm \\
        --holdout-participants CO001,FL001 \\
        --extraction-profile free_living
"""

import argparse
from pathlib import Path

from src.fall_risk_assessment.model_bundle import ExtractionConfig
from src.fall_risk_assessment.training import FallRiskModelTrainer
from src.classification.models.model_factory import available_model_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a deployable fall-risk model bundle."
    )
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=f"Comma-separated model families (default all: {available_model_names()})",
    )
    parser.add_argument(
        "--holdout-participants",
        type=str,
        default="",
        help="Comma-separated participant IDs excluded from training entirely.",
    )
    parser.add_argument("--extraction-backend", type=str, default="skdh")
    parser.add_argument("--extraction-profile", type=str, default="free_living")
    parser.add_argument("--stride-catalog", type=str, default="skdh")
    parser.add_argument("--epoch-window-seconds", type=float, default=8.0)
    parser.add_argument("--epoch-overlap-seconds", type=float, default=2.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fast-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    holdouts = [
        pid.strip() for pid in args.holdout_participants.split(",") if pid.strip()
    ]
    trainer = FallRiskModelTrainer(
        sqlite_db_path=args.sqlite_db_path,
        output_dir=args.output_dir,
        extraction_config=ExtractionConfig(
            backend=args.extraction_backend,
            profile=args.extraction_profile,
            epoch_window_seconds=args.epoch_window_seconds,
            epoch_overlap_seconds=args.epoch_overlap_seconds,
            stride_catalog=args.stride_catalog,
        ),
        model_names=model_names or None,
        holdout_participant_ids=holdouts,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        fast_mode=args.fast_mode,
    )
    result = trainer.train()
    print(f"[fall-risk-train] champion: {result.bundle.model_name}")
    print(f"[fall-risk-train] bundle: {result.bundle_dir}")
    print(f"[fall-risk-train] CV artifact: {result.evaluation_artifact_path}")


if __name__ == "__main__":
    main()
