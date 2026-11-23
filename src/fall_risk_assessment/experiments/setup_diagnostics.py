"""Diagnostics: is the fall-risk setup under-using the information in LTMM?

Fast, pre-specified analyses behind docs/SYSTEMATIC_REVIEW.md:

  1. Label audit against the LTMM clinical spreadsheet.
  2. Clinical benchmark: a-priori-oriented univariate AUCs of the LTMM clinical
     and performance measures, on the same participants as the gait models.
  3. Criterion validity: extracted gait speed (3-day free-living and lab walk)
     against the stopwatch lab gait speed in the clinical spreadsheet.
  4. Univariate screen of every gait feature with a family-wise
     (max-|AUC - 0.5|) label-permutation test.
  5. Pre-specified low-dimensional models under the same 5x5
     participant-stratified repeated CV as the production evaluation, with
     participant-bootstrap CIs and label-permutation p-values.
  6. Sensitivity analyses: clean contrast (non-fallers with zero falls) and
     exclusion of participants with implausibly little detected gait.

Every model is fixed before seeing results — median imputation,
standardization, L2 logistic regression (C=1), balanced class weights, no
feature selection, no tuning — so each estimate is free of selection
optimism. Comparisons *across* the set are descriptive.

Run:
    python -m src.fall_risk_assessment.experiments.setup_diagnostics \\
        --day3-db .../sql_db_indexes/ltmm/index.db \\
        --lab-db .../sql_db_indexes/lab_walks/index.db \\
        --clinical-xlsx .../raw_data/ltmm/ClinicalDemogData_COFL.xlsx \\
        --output-dir .../models/ltmm \\
        --holdout-participants CO002,CO010,FL004,FL010
"""

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classification.evaluation.cross_validation import (
    iter_participant_folds,
    resolve_n_splits,
)
from src.fall_risk_assessment.statistical_validation import bootstrap_metric_ci

# Spreadsheet column -> canonical name. "6 Months Fall" is deliberately absent:
# it is a subset of the past-year falls that define the label.
_CLINICAL_COLUMNS = {
    "Age": "age",
    "TUG": "tug_s",
    "FSST": "fsst_s",
    "BERG": "berg",
    "DGI": "dgi",
    "ABC Tot %": "abc_pct",
    "base(velocity)": "lab_gait_speed",
    "s3(velocity)": "dual_task_gait_speed",
    "GDS": "gds",
    "PASE": "pase",
    "MoCa": "moca",
    "MMSE": "mmse",
    "TMTb": "tmt_b_s",
    "SF-36": "sf36",
    " yr almost": "near_falls_year",
    "Year Fall ": "falls_past_year",
}

# A-priori direction: +1 when a higher value indicates higher fall risk.
CLINICAL_DIRECTIONS = {
    "age": +1,
    "female": +1,
    "tug_s": +1,
    "fsst_s": +1,
    "berg": -1,
    "dgi": -1,
    "abc_pct": -1,
    "lab_gait_speed": -1,
    "dual_task_gait_speed": -1,
    "gds": +1,
    "pase": -1,
    "moca": -1,
    "mmse": -1,
    "tmt_b_s": +1,
    "sf36": -1,
    "near_falls_year": +1,
}

# Measures that a past fall can itself change (fear of falling, mood, quality
# of life, near-fall recall): in a retrospective design their AUC partly
# reflects consequences of the label, not risk factors for it.
REVERSE_CAUSATION_PRONE = {"abc_pct", "near_falls_year", "gds", "sf36"}

# Fixed before looking at any result: the most-replicated gait markers of fall
# risk in the literature that the skdh stride catalog provides.
PRESPECIFIED_GAIT = [
    "GAIT_SPEED",
    "STRIDE_TIME",
    "DOUBLE_SUPPORT",
    "HARMONIC_RATIO_V",
    "INTRA_STRIDE_COVARIANCE_V",
    "STRIDE_SPARC",
]
PRESPECIFIED_CLINICAL = ["age", "female", "tug_s", "berg", "dgi", "lab_gait_speed"]
MIN_PLAUSIBLE_STRIDES = 500  # over 3 days; fewer implies failed gait detection


def normalize_participant_id(raw) -> str:
    return str(raw).strip().upper().replace("-", "").replace(" ", "")


def parse_clinical_sheets(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Controls/Fallers sheets -> one numeric table indexed by participant id."""
    frames = []
    for sheet_name, group in (("Controls", 0), ("Fallers", 1)):
        df = sheets[sheet_name]
        gender_col = next(c for c in df.columns if str(c).lower().startswith("gender"))
        out = pd.DataFrame({"participant_id": df["#"].map(normalize_participant_id)})
        out["group_faller"] = group
        out["female"] = pd.to_numeric(df[gender_col], errors="coerce").to_numpy()
        for source, target in _CLINICAL_COLUMNS.items():
            values = df[source] if source in df.columns else pd.Series(np.nan, index=df.index)
            out[target] = pd.to_numeric(values, errors="coerce").to_numpy()
        frames.append(out)
    return pd.concat(frames, ignore_index=True).set_index("participant_id")


def load_clinical_table(xlsx_path: Path) -> pd.DataFrame:
    return parse_clinical_sheets(pd.read_excel(xlsx_path, sheet_name=None))


def oriented_auc(y: np.ndarray, x: np.ndarray, direction: int = 1) -> Dict[str, float]:
    """AUC of ``direction * x`` for predicting y, on rows where x is finite."""
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    n = int(mask.sum())
    if n == 0 or len(np.unique(y[mask])) < 2:
        return {"auc": float("nan"), "n": n}
    return {"auc": float(roc_auc_score(y[mask], direction * x[mask])), "n": n}


def _column_ranks(X: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    return np.apply_along_axis(rankdata, 0, filled)


def max_stat_screen(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    n_permutations: int = 2000,
    random_state: int = 42,
) -> Dict[str, object]:
    """Univariate AUC per feature with uncorrected and family-wise p-values.

    The family-wise p-value of feature j is the share of label permutations
    whose *largest* |AUC - 0.5| over all features reaches feature j's
    observed |AUC - 0.5| (Westfall-Young max-T), controlling the chance of
    any false discovery across the whole feature set.
    """
    y = np.asarray(y, dtype=int)
    ranks = _column_ranks(np.asarray(X, dtype=float))
    n1 = int(y.sum())
    n0 = len(y) - n1
    offset = n1 * (n1 + 1) / 2.0

    def aucs(positive_mask: np.ndarray) -> np.ndarray:
        return (ranks[positive_mask].sum(axis=0) - offset) / (n1 * n0)

    observed = aucs(y == 1)
    observed_dev = np.abs(observed - 0.5)
    rng = np.random.default_rng(random_state)
    null_max = np.empty(n_permutations)
    exceed = np.zeros(len(observed))
    for b in range(n_permutations):
        mask = np.zeros(len(y), dtype=bool)
        mask[rng.choice(len(y), size=n1, replace=False)] = True
        dev = np.abs(aucs(mask) - 0.5)
        null_max[b] = np.nanmax(dev)
        exceed += np.nan_to_num(dev, nan=-1.0) >= observed_dev - 1e-12
    p_uncorrected = (1 + exceed) / (n_permutations + 1)
    p_familywise = np.array(
        [(1 + np.sum(null_max >= d - 1e-12)) / (n_permutations + 1) for d in observed_dev]
    )
    order = np.argsort(-np.nan_to_num(observed_dev, nan=-1.0))
    return {
        "n_features": len(names),
        "n_permutations": n_permutations,
        "max_abs_deviation": float(np.nanmax(observed_dev)),
        "familywise_p_of_max": float(p_familywise[order[0]]),
        "features": [
            {
                "feature": str(names[j]),
                "auc": float(observed[j]),
                "p_uncorrected": float(p_uncorrected[j]),
                "p_familywise": float(p_familywise[j]),
            }
            for j in order
        ],
    }


def fixed_model():
    """The single pre-specified model: no selection, no tuning."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
    )


def participant_averaged_oof(
    X: np.ndarray, y: np.ndarray, n_splits: int, n_repeats: int, random_state: int
) -> np.ndarray:
    """Out-of-fold probabilities averaged over repeats, one per participant."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    sums = np.zeros(len(y))
    counts = np.zeros(len(y))
    splits = resolve_n_splits(y, desired=n_splits)
    for train_idx, test_idx in iter_participant_folds(y, splits, n_repeats, random_state):
        model = fixed_model().fit(X[train_idx], y[train_idx])
        sums[test_idx] += model.predict_proba(X[test_idx])[:, 1]
        counts[test_idx] += 1
    return sums / np.maximum(counts, 1)


def univariate_auc_deviation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|AUC - 0.5| per column (NaNs median-filled for ranking)."""
    y = np.asarray(y, dtype=int)
    ranks = _column_ranks(np.asarray(X, dtype=float))
    n1 = int(y.sum())
    n0 = len(y) - n1
    auc = (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return np.abs(np.nan_to_num(auc, nan=0.5) - 0.5)


def _select_k_columns(X_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-univariate_auc_deviation(X_train, y_train), kind="stable")[:k]


def select_k_best_oof(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    k: int = 1,
) -> np.ndarray:
    """Like participant_averaged_oof, but the k most discriminative features
    (univariate |AUC - 0.5|) are chosen inside every training fold — the
    honest, nested version of "use the best single feature"."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    sums = np.zeros(len(y))
    counts = np.zeros(len(y))
    splits = resolve_n_splits(y, desired=n_splits)
    for train_idx, test_idx in iter_participant_folds(y, splits, n_repeats, random_state):
        cols = _select_k_columns(X[train_idx], y[train_idx], k)
        model = fixed_model().fit(X[train_idx][:, cols], y[train_idx])
        sums[test_idx] += model.predict_proba(X[test_idx][:, cols])[:, 1]
        counts[test_idx] += 1
    return sums / np.maximum(counts, 1)


def selection_frequencies(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    k: int,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, int]:
    """How often each feature is picked by select_k_best_oof's inner step."""
    y = np.asarray(y, dtype=int)
    tally: Dict[str, int] = {}
    splits = resolve_n_splits(y, desired=n_splits)
    for train_idx, _ in iter_participant_folds(y, splits, n_repeats, random_state):
        for col in _select_k_columns(np.asarray(X)[train_idx], y[train_idx], k):
            tally[str(names[col])] = tally.get(str(names[col]), 0) + 1
    return dict(sorted(tally.items(), key=lambda kv: -kv[1]))


def _null_auc(X, shuffled, n_splits, n_repeats, random_state, oof_fn) -> float:
    return float(roc_auc_score(shuffled, oof_fn(X, shuffled, n_splits, n_repeats, random_state)))


def evaluate_fixed_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    n_bootstrap: int = 2000,
    n_permutations: int = 200,
    n_jobs: int = -1,
    oof_fn: Callable = participant_averaged_oof,
) -> Dict[str, float]:
    """Repeated-CV AUC, participant-bootstrap CI, and permutation p-value.

    ``oof_fn`` produces the participant-averaged out-of-fold probabilities;
    the permutation null reruns the *same* procedure (including any inner
    feature selection) on shuffled labels.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    p = oof_fn(X, y, n_splits, n_repeats, random_state)
    ci = bootstrap_metric_ci(y, p, n_bootstrap=n_bootstrap, random_state=random_state)
    result = {
        "n": int(len(y)),
        "n_fallers": int(y.sum()),
        "n_features": int(X.shape[1]),
        "auc": float(roc_auc_score(y, p)),
        "ci_lower": float(ci["ci_lower"]),
        "ci_upper": float(ci["ci_upper"]),
    }
    if n_permutations:
        rng = np.random.default_rng(random_state)
        shuffles = [rng.permutation(y) for _ in range(n_permutations)]
        null = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_null_auc)(X, shuffled, n_splits, n_repeats, random_state, oof_fn)
                for shuffled in shuffles
            )
        )
        result["permutation_p"] = float((1 + np.sum(null >= result["auc"])) / (n_permutations + 1))
        result["null_auc_mean"] = float(null.mean())
    return result


def _names(dataset) -> List[str]:
    return [
        str(name).replace("FeatureType.", "")
        for name in list(dataset.stride_feature_names) + list(dataset.epoch_feature_names)
    ]


def _columns(dataset, wanted: Sequence[str]) -> np.ndarray:
    names = _names(dataset)
    return dataset.X[:, [names.index(feature) for feature in wanted]]


def _stride_time_std(dataset_mean_std) -> np.ndarray:
    """Participant-level std of stride time from a mean_std dataset."""
    n_stride = len(dataset_mean_std.stride_feature_names) // 2
    base = [str(n).replace("FeatureType.", "") for n in dataset_mean_std.stride_feature_names[:n_stride]]
    return dataset_mean_std.X[:, n_stride + base.index("STRIDE_TIME")]


def _subset(mask: np.ndarray, *arrays: np.ndarray):
    return [np.asarray(a)[mask] for a in arrays]


def run_diagnostics(
    day3,
    day3_mean_std,
    lab,
    clinical: pd.DataFrame,
    stride_counts: Mapping[str, int],
    n_permutations: int = 200,
    random_state: int = 42,
) -> Dict[str, object]:
    """All analyses on prebuilt participant datasets (holdouts already removed)."""
    report: Dict[str, object] = {}
    ids = list(day3.participant_ids)
    y = np.asarray(day3.y, dtype=int)
    clin = clinical.reindex(ids)

    # 1. Label audit
    falls = clin["falls_past_year"].to_numpy()
    report["label_audit"] = {
        "n_participants": len(ids),
        "n_fallers": int(y.sum()),
        "label_matches_clinical_group": int(np.sum(clin["group_faller"].to_numpy() == y)),
        "non_fallers_with_one_fall": int(np.sum((y == 0) & (falls == 1))),
        "fallers_with_fewer_than_two_falls": [
            pid for pid, yy, f in zip(ids, y, falls) if yy == 1 and np.isfinite(f) and f < 2
        ],
    }

    # 2. Clinical benchmark (univariate, a-priori orientation)
    clinical_rows = []
    for measure, direction in CLINICAL_DIRECTIONS.items():
        x = clin[measure].to_numpy(dtype=float)
        stats = oriented_auc(y, x, direction)
        mask = np.isfinite(x)
        if mask.sum() > 10 and len(np.unique(y[mask])) == 2:
            ci = bootstrap_metric_ci(y[mask], direction * x[mask], random_state=random_state)
            stats.update(ci_lower=float(ci["ci_lower"]), ci_upper=float(ci["ci_upper"]))
        stats.update(measure=measure, reverse_causation_prone=measure in REVERSE_CAUSATION_PRONE)
        clinical_rows.append(stats)
    report["clinical_univariate"] = sorted(clinical_rows, key=lambda r: -np.nan_to_num(r["auc"]))

    # 3. Criterion validity of extracted gait speed vs stopwatch lab speed
    lab_speed = clin["lab_gait_speed"].to_numpy(dtype=float)
    validity = {}
    for label, dataset in (("day3_free_living", day3), ("lab_walk", lab)):
        lab_ref = clinical.reindex(dataset.participant_ids)["lab_gait_speed"].to_numpy(dtype=float)
        for feature in ("GAIT_SPEED", "GAIT_SPEED_M1", "CADENCE"):
            x = _columns(dataset, [feature])[:, 0]
            mask = np.isfinite(x) & np.isfinite(lab_ref)
            rho, pval = spearmanr(x[mask], lab_ref[mask])
            validity[f"{label}:{feature}"] = {
                "spearman_rho": float(rho),
                "p": float(pval),
                "n": int(mask.sum()),
                "median_extracted": float(np.median(x[mask])),
                "median_lab_reference": float(np.median(lab_ref[mask])),
            }
    report["criterion_validity_vs_lab_speed"] = validity

    # 4. Univariate screens of all gait features
    report["gait_univariate_screen"] = {
        "day3_free_living": max_stat_screen(day3.X, y, _names(day3), random_state=random_state),
        "lab_walk": max_stat_screen(lab.X, lab.y, _names(lab), random_state=random_state),
    }
    for screen in report["gait_univariate_screen"].values():
        screen["features"] = screen["features"][:12]

    # 5. Pre-specified models under the production CV protocol
    def run(name, X, yy, **meta):
        result = evaluate_fixed_model(
            X, yy, random_state=random_state, n_permutations=n_permutations
        )
        result.update(model=name, **meta)
        print(
            f"[diagnostics] {name:45s} n={result['n']:3d} AUC={result['auc']:.3f} "
            f"({result['ci_lower']:.3f}-{result['ci_upper']:.3f}) "
            f"p={result.get('permutation_p', float('nan')):.3f}"
        )
        return result

    clin_pre = clin[PRESPECIFIED_CLINICAL].to_numpy(dtype=float)
    gait_pre = _columns(day3, PRESPECIFIED_GAIT)
    speed = _columns(day3, ["GAIT_SPEED"])
    variability = np.column_stack([gait_pre, _stride_time_std(day3_mean_std)])
    lab_y = np.asarray(lab.y, dtype=int)
    lab_clin = clinical.reindex(lab.participant_ids)[PRESPECIFIED_CLINICAL].to_numpy(dtype=float)

    models = [
        run("day3: gait speed only", speed, y, dataset="day3"),
        run("day3: pre-specified gait set (6)", gait_pre, y, dataset="day3"),
        run("day3: gait set + stride-time SD (7)", variability, y, dataset="day3"),
        run("day3: all 76 features, ridge LR", day3.X, y, dataset="day3"),
        run("lab: gait speed only", _columns(lab, ["GAIT_SPEED"]), lab_y, dataset="lab"),
        run("lab: pre-specified gait set (6)", _columns(lab, PRESPECIFIED_GAIT), lab_y, dataset="lab"),
        run("lab: all 76 features, ridge LR", lab.X, lab_y, dataset="lab"),
        run("clinical: age + sex", clin[["age", "female"]].to_numpy(dtype=float), y, dataset="clinical"),
        run("clinical: pre-specified (6)", clin_pre, y, dataset="clinical"),
        run(
            "clinical: pre-specified + ABC (reverse-causation-prone)",
            np.column_stack([clin_pre, clin["abc_pct"].to_numpy(dtype=float)]),
            y,
            dataset="clinical",
        ),
        run("clinical (6) + day3 gait set (6)", np.column_stack([clin_pre, gait_pre]), y, dataset="combined"),
        run(
            "clinical (6) + lab gait set (6)",
            np.column_stack([lab_clin, _columns(lab, PRESPECIFIED_GAIT)]),
            lab_y,
            dataset="combined",
        ),
    ]
    report["prespecified_models"] = models

    # 5b. Nested data-driven selection of the k best features (selection inside
    # every training fold and every permutation), and a matched-participant
    # comparison of free-living vs lab-walk gait speed.
    def run_select(name, X, yy, names, k, **meta):
        result = evaluate_fixed_model(
            X,
            yy,
            random_state=random_state,
            n_permutations=n_permutations,
            oof_fn=partial(select_k_best_oof, k=k),
        )
        result.update(
            model=name,
            k=k,
            selection_frequencies=dict(
                list(selection_frequencies(X, yy, names, k, random_state=random_state).items())[:6]
            ),
            **meta,
        )
        print(
            f"[diagnostics] {name:45s} n={result['n']:3d} AUC={result['auc']:.3f} "
            f"({result['ci_lower']:.3f}-{result['ci_upper']:.3f}) p={result['permutation_p']:.3f}"
        )
        return result

    shared = [pid for pid in lab.participant_ids if pid in set(ids)]
    day3_rows = [ids.index(pid) for pid in shared]
    lab_rows = [list(lab.participant_ids).index(pid) for pid in shared]
    y_shared = y[day3_rows]
    report["nested_selection_and_matched"] = [
        run_select("day3: nested best-1 of 76", day3.X, y, _names(day3), 1, dataset="day3"),
        run_select("day3: nested best-3 of 76", day3.X, y, _names(day3), 3, dataset="day3"),
        run_select("lab: nested best-1 of 76", lab.X, lab_y, _names(lab), 1, dataset="lab"),
        run_select("lab: nested best-3 of 76", lab.X, lab_y, _names(lab), 3, dataset="lab"),
        run("matched: day3 gait speed", speed[day3_rows], y_shared, dataset="day3", subset="lab_participants"),
        run(
            "matched: lab gait speed",
            _columns(lab, ["GAIT_SPEED"])[lab_rows],
            np.asarray(lab.y)[lab_rows],
            dataset="lab",
            subset="lab_participants",
        ),
        run(
            "matched: stopwatch lab speed (clinical)",
            clin["lab_gait_speed"].to_numpy(dtype=float)[day3_rows][:, None],
            y_shared,
            dataset="clinical",
            subset="lab_participants",
        ),
    ]

    # 6. Sensitivity analyses
    clean = (y == 1) | (falls == 0)
    counts = np.array([stride_counts.get(pid, 0) for pid in ids])
    enough = counts >= MIN_PLAUSIBLE_STRIDES
    report["sensitivity"] = {
        "clean_contrast_excluded": [pid for pid, keep in zip(ids, clean) if not keep],
        "low_gait_excluded": {pid: int(c) for pid, c, keep in zip(ids, counts, enough) if not keep},
        "models": [
            run("clean: day3 gait set (6)", *_subset(clean, gait_pre, y), dataset="day3", subset="clean_contrast"),
            run("clean: day3 all 76, ridge", *_subset(clean, day3.X, y), dataset="day3", subset="clean_contrast"),
            run("clean: clinical pre-specified (6)", *_subset(clean, clin_pre, y), dataset="clinical", subset="clean_contrast"),
            run("enough-gait: day3 gait set (6)", *_subset(enough, gait_pre, y), dataset="day3", subset="min_strides"),
            run("enough-gait: day3 all 76, ridge", *_subset(enough, day3.X, y), dataset="day3", subset="min_strides"),
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day3-db", type=Path, required=True)
    parser.add_argument("--lab-db", type=Path, required=True)
    parser.add_argument("--clinical-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-participants", type=str, default="")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    holdouts = [p.strip() for p in args.holdout_participants.split(",") if p.strip()]

    # Imported here so the pure helpers above stay importable without the DB stack.
    from src.classification.data.classification_dataset_builder import (
        ClassificationDatasetBuilder,
    )
    from src.database_manager.database_generator import DatabaseGenerator
    from src.fall_risk_assessment.experiments.aggregation_comparison import (
        build_variant_dataset,
    )

    def load(db_path: Path):
        builder = ClassificationDatasetBuilder(
            db_manager=DatabaseGenerator().generate_database(db_path, validate=True)
        )
        return builder, builder.build(stride_catalog="skdh")

    day3_builder, day3_sample = load(args.day3_db)
    lab_builder, lab_sample = load(args.lab_db)
    day3 = build_variant_dataset(day3_builder, "mean", holdouts, sample=day3_sample)
    day3_mean_std = build_variant_dataset(day3_builder, "mean_std", holdouts, sample=day3_sample)
    lab = build_variant_dataset(lab_builder, "mean", holdouts, sample=lab_sample)
    groups, counts = np.unique(day3_sample.stride.groups, return_counts=True)
    stride_counts = {str(g): int(c) for g, c in zip(groups, counts)}
    print(
        f"[diagnostics] day3 {day3.n_participants} participants, "
        f"lab {lab.n_participants} participants"
    )

    report = run_diagnostics(
        day3,
        day3_mean_std,
        lab,
        load_clinical_table(args.clinical_xlsx),
        stride_counts,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    report.update(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        day3_db=str(args.day3_db),
        lab_db=str(args.lab_db),
        holdout_participant_ids=holdouts,
        protocol={
            "cv": "5-fold x 5 repeats, participant-stratified",
            "model": "median impute + standardize + L2 logistic regression (C=1, balanced)",
            "auc": "participant-averaged out-of-fold probabilities",
            "ci": "percentile bootstrap over participants (2000)",
            "permutations": args.n_permutations,
            "random_state": args.random_state,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = args.output_dir / f"setup_diagnostics_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"[diagnostics] report: {path}")


if __name__ == "__main__":
    main()
