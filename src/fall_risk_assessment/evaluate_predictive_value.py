"""CLI: unbiased performance estimate + statistical inference for the pipeline.

Produces a ``predictive_value_report.json`` answering, with uncertainty:
"does this pipeline predict faller status better than chance?"

  - Nested family-selection CV (selection inside every outer fold) ->
    out-of-fold ROC-AUC / PR-AUC free of winner's-curse optimism
  - Participant-level percentile-bootstrap confidence intervals
  - Permutation test of the identical pipeline on shuffled labels -> p-value

Example:
    python -m src.fall_risk_assessment.evaluate_predictive_value \\
        --sqlite-db-path .../sql_db_indexes/ltmm/index.db \\
        --output-dir .../models/ltmm \\
        --holdout-participants CO002,CO010,FL004,FL010 \\
        --n-permutations 200
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.fall_risk_assessment.model_bundle import ExtractionConfig
from src.fall_risk_assessment.statistical_validation import (
    bootstrap_metric_ci,
    nested_family_selection_cv,
    permutation_test,
)
from src.fall_risk_assessment.training import FallRiskModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the pipeline's predictive value with nested CV, "
        "bootstrap CIs, and a permutation test."
    )
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-participants", type=str, default="")
    parser.add_argument("--stride-catalog", type=str, default="skdh")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip the permutation test (nested CV + bootstrap only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holdouts = [
        pid.strip() for pid in args.holdout_participants.split(",") if pid.strip()
    ]

    # Reuse the trainer's dataset assembly + holdout exclusion so the evaluated
    # population exactly matches the trained model's population.
    trainer = FallRiskModelTrainer(
        sqlite_db_path=args.sqlite_db_path,
        output_dir=args.output_dir,
        extraction_config=ExtractionConfig(stride_catalog=args.stride_catalog),
        holdout_participant_ids=holdouts,
        random_state=args.random_state,
    )
    full_dataset, _ = trainer._build_dataset()
    dataset, excluded = trainer._exclude_holdouts(full_dataset)
    print(
        f"[predictive-value] {dataset.n_participants} participants "
        f"({int(np.sum(dataset.y == 1))} fallers / "
        f"{int(np.sum(dataset.y == 0))} non-fallers), "
        f"{dataset.n_features} features, {len(excluded)} held out"
    )

    print("[predictive-value] running nested family-selection CV (full grids)...")
    nested = nested_family_selection_cv(
        dataset,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
        fast_mode=False,
        verbose=True,
    )
    participant_y, participant_p = nested.participant_level_probabilities(
        dataset.n_participants
    )
    roc_ci = bootstrap_metric_ci(
        participant_y,
        participant_p,
        metric=roc_auc_score,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )
    pr_ci = bootstrap_metric_ci(
        participant_y,
        participant_p,
        metric=average_precision_score,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )
    print(
        f"[predictive-value] nested OOF ROC-AUC={nested.roc_auc:.3f} "
        f"(participant-averaged {roc_ci['point_estimate']:.3f}, "
        f"95% CI {roc_ci['ci_lower']:.3f}-{roc_ci['ci_upper']:.3f})"
    )

    permutation = None
    if not args.skip_permutation:
        print(
            f"[predictive-value] permutation test "
            f"({args.n_permutations} permutations, matched fast procedure)..."
        )
        permutation = permutation_test(
            dataset,
            n_permutations=args.n_permutations,
            n_splits=args.n_splits,
            inner_splits=args.inner_splits,
            random_state=args.random_state,
            fast_mode=True,
            verbose=True,
        )
        print(
            f"[predictive-value] p={permutation['p_value']:.4f} "
            f"(observed {permutation['observed_auc']:.3f} vs null "
            f"{permutation['null_auc_mean']:.3f}±{permutation['null_auc_std']:.3f})"
        )

    prevalence = float(np.mean(participant_y))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sqlite_db_path": str(Path(args.sqlite_db_path).resolve()),
        "population": {
            "n_participants": dataset.n_participants,
            "n_fallers": int(np.sum(dataset.y == 1)),
            "n_non_fallers": int(np.sum(dataset.y == 0)),
            "prevalence": prevalence,
            "holdout_participant_ids": excluded,
            "n_features": dataset.n_features,
        },
        "nested_cv": nested.summary(),
        "roc_auc_bootstrap": roc_ci,
        "pr_auc_bootstrap": {**pr_ci, "chance_level": prevalence},
        "permutation_test": permutation,
        "methodology": {
            "outer_cv": f"{nested.n_splits}-fold x {nested.n_repeats} repeats, "
            "participant-stratified",
            "family_selection": "inner CV on outer-train folds only (nested; "
            "no winner's curse)",
            "hyperparameter_tuning": "grid search inside each family fit "
            "(outer-train data only)",
            "confidence_intervals": "percentile bootstrap over participants, "
            "participant-averaged OOF probabilities",
            "permutation_test": "label shuffle at participant level; identical "
            "nested pipeline (fast grids) for observed and null statistics",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"predictive_value_report_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[predictive-value] report: {report_path}")


if __name__ == "__main__":
    main()
