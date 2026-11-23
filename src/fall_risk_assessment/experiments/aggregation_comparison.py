"""Experiment: does dispersion-aware aggregation beat mean-only aggregation?

The methodology review (docs/METHODOLOGY_REVIEW.md §3) identifies mean-only
participant aggregation as the pipeline's most likely feature-side ceiling:
gait *variability* is a leading fall-risk marker, and a nanmean collapses it.

This experiment compares, under the identical nested family-selection CV:
  A. mean aggregation (production pipeline, ~76 features)
  B. mean + std aggregation (~152 features)
  C. mean + std + IQR aggregation (~228 features)

Each variant is evaluated with `nested_family_selection_cv`, so added
dimensionality pays its own way honestly — if dispersion features only add
noise at this n, variant B/C will score worse, not better.

Run:
    python -m src.fall_risk_assessment.experiments.aggregation_comparison \\
        --sqlite-db-path .../index.db --output-dir .../models/ltmm \\
        --holdout-participants CO002,CO010,FL004,FL010
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.classification.data.classification_dataset import (
    BasisSampleDataset,
    EarlyFusionParticipantDataset,
)
from src.classification.data.classification_dataset_builder import (
    ClassificationDatasetBuilder,
)
from src.database_manager.database_generator import DatabaseGenerator
from src.fall_risk_assessment.statistical_validation import (
    bootstrap_metric_ci,
    nested_family_selection_cv,
)


def _nan_iqr(block: np.ndarray) -> np.ndarray:
    return np.nanquantile(block, 0.75, axis=0) - np.nanquantile(block, 0.25, axis=0)


_AGGREGATORS = {
    "mean": [lambda block: np.nanmean(block, axis=0)],
    "mean_std": [
        lambda block: np.nanmean(block, axis=0),
        lambda block: np.nanstd(block, axis=0),
    ],
    "mean_std_iqr": [
        lambda block: np.nanmean(block, axis=0),
        lambda block: np.nanstd(block, axis=0),
        _nan_iqr,
    ],
}


def _aggregate(
    sample_dataset: BasisSampleDataset,
    participant_order: List[str],
    variant: str,
) -> Dict[str, np.ndarray]:
    """One row per participant: concatenated aggregate blocks for a basis."""
    label_map = sample_dataset.participant_labels()
    rows, labels, kept = [], [], []
    for participant_id in participant_order:
        mask = sample_dataset.groups == participant_id
        if not np.any(mask):
            continue
        block = sample_dataset.X[mask]
        with np.errstate(all="ignore"):
            parts = [aggregator(block) for aggregator in _AGGREGATORS[variant]]
        rows.append(np.concatenate(parts))
        labels.append(int(label_map[participant_id]))
        kept.append(participant_id)
    return {
        "X": np.vstack(rows),
        "y": np.asarray(labels, dtype=int),
        "participant_ids": kept,
    }


def build_variant_dataset(
    builder: ClassificationDatasetBuilder,
    variant: str,
    holdouts: List[str],
    stride_catalog: str = "skdh",
    sample=None,
) -> EarlyFusionParticipantDataset:
    """Participant-level dataset for one aggregation variant.

    ``sample`` lets callers that build several variants reuse one loaded
    ClassificationDataset instead of re-reading every feature payload.
    """
    if sample is None:
        sample = builder.build(stride_catalog=stride_catalog)
    common = [pid for pid in sample.common_participant_ids() if pid not in holdouts]
    stride = _aggregate(sample.stride, common, variant)
    epoch = _aggregate(sample.epoch, common, variant)
    if stride["participant_ids"] != epoch["participant_ids"]:
        raise ValueError("Stride/epoch participant order mismatch in experiment.")
    # Feature-name metadata is cosmetic for this experiment (metrics only);
    # reuse base names for the dataclass contract.
    n_copies = len(_AGGREGATORS[variant])
    return EarlyFusionParticipantDataset(
        X=np.hstack([stride["X"], epoch["X"]]),
        y=stride["y"],
        participant_ids=stride["participant_ids"],
        stride_feature_names=list(sample.stride.feature_names) * n_copies,
        epoch_feature_names=list(sample.epoch.feature_names) * n_copies,
        aggregation=variant,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-participants", type=str, default="")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--fast-mode", action="store_true",
        help="Use reduced hyperparameter grids for a quicker comparison.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(_AGGREGATORS),
        help=f"Comma-separated subset of {list(_AGGREGATORS)} to run.",
    )
    parser.add_argument(
        "--resume-report",
        type=Path,
        default=None,
        help="Partial report from an interrupted run: variants it already "
        "contains are skipped and new results are written back into it.",
    )
    args = parser.parse_args()
    holdouts = [p.strip() for p in args.holdout_participants.split(",") if p.strip()]
    variants = parse_variants(args.variants)
    cv_config = {
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "fast_mode": args.fast_mode,
        "random_state": args.random_state,
    }

    if args.resume_report is not None:
        report_path = Path(args.resume_report)
        report = load_resumable_report(report_path, holdouts, cv_config)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = args.output_dir / f"aggregation_comparison_{timestamp}.json"
        report = {
            "holdout_participant_ids": holdouts,
            "cv": cv_config,
            "requested_variants": variants,
            "variants": {},
        }
    results: Dict[str, object] = report["variants"]
    pending = [variant for variant in variants if variant not in results]
    for variant in variants:
        if variant in results:
            print(f"[aggregation-exp] {variant}: already in report, skipping")

    db_manager = DatabaseGenerator().generate_database(
        args.sqlite_db_path, validate=True
    )
    builder = ClassificationDatasetBuilder(db_manager=db_manager)

    for variant in pending:
        dataset = build_variant_dataset(builder, variant, holdouts)
        print(
            f"[aggregation-exp] {variant}: {dataset.n_participants} participants, "
            f"{dataset.n_features} features"
        )
        nested = nested_family_selection_cv(
            dataset,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
            fast_mode=args.fast_mode,
            verbose=True,
            log=lambda message, variant=variant: print(f"[{variant}] {message}"),
        )
        y, p = nested.participant_level_probabilities(dataset.n_participants)
        ci = bootstrap_metric_ci(y, p, random_state=args.random_state)
        results[variant] = {
            "n_features": dataset.n_features,
            "nested_oof_roc_auc": nested.roc_auc,
            "participant_averaged_roc_auc": ci["point_estimate"],
            "roc_auc_ci_95": [ci["ci_lower"], ci["ci_upper"]],
            "selected_family_counts": nested.selected_family_counts,
        }
        print(
            f"[aggregation-exp] {variant}: nested AUC={nested.roc_auc:.3f} "
            f"(95% CI {ci['ci_lower']:.3f}-{ci['ci_upper']:.3f})"
        )
        # Persist after every variant so an interrupted run keeps finished work.
        write_report(report_path, report, variants)
        print(f"[aggregation-exp] saved {variant} -> {report_path}")

    write_report(report_path, report, variants)
    print(f"[aggregation-exp] report ({report['status']}): {report_path}")


def parse_variants(raw: str) -> List[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in _AGGREGATORS]
    if unknown or not variants:
        raise SystemExit(
            f"Unknown or empty --variants {unknown or raw!r}; "
            f"choose from {list(_AGGREGATORS)}"
        )
    return variants


def load_resumable_report(
    report_path: Path, holdouts: List[str], cv_config: Dict[str, object]
) -> Dict[str, object]:
    """Load a partial report, refusing to mix results from different setups."""
    with Path(report_path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("holdout_participant_ids") != holdouts:
        raise SystemExit(
            "--resume-report holdouts differ from --holdout-participants; "
            "results would not be comparable."
        )
    saved_cv = report.get("cv", {})
    mismatched = {
        key: (saved_cv.get(key), value)
        for key, value in cv_config.items()
        if key in saved_cv and saved_cv[key] != value
    }
    if mismatched:
        raise SystemExit(
            f"--resume-report CV settings differ (saved, requested): {mismatched}"
        )
    report.setdefault("variants", {})
    return report


def write_report(
    report_path: Path, report: Dict[str, object], variants: List[str]
) -> None:
    done = report["variants"]
    report["status"] = (
        "complete" if all(variant in done for variant in variants) else "partial"
    )
    report["requested_variants"] = sorted(
        set(report.get("requested_variants", [])) | set(variants),
        key=list(_AGGREGATORS).index,
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp_path = Path(report_path).with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    tmp_path.replace(report_path)


if __name__ == "__main__":
    main()
