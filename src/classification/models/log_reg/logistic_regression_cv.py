import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.classification.feature_preprocessor.feature_preprocessor import (
    FeaturePreprocessor,
    PreparedDataset,
)
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.database_manager.database_generator import DatabaseGenerator
from src.database_manager.database_manager import DatabaseManager
from src.identifiers.feature.aggregate_feature_identifier import AggregateFeatureIdentifier
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


def _json_safe(value):
    """Convert numpy/scalar/NaN values to JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _build_feature_definition_records(feature_names) -> List[Dict[str, str]]:
    """
    Convert feature-name tuples into explicit model feature definitions.

    Each record maps a model column to:
      - raw_feature_type value
      - descriptive_stat_type value
    """
    records: List[Dict[str, str]] = []
    for index, (raw_feature_type, stat_type) in enumerate(feature_names):
        records.append(
            {
                "feature_index": index,
                "raw_feature_type": raw_feature_type.value,
                "descriptive_stat_type": stat_type.value,
                "feature_key": f"{raw_feature_type.value}::{stat_type.value}",
            }
        )
    return records


def _default_results_output_path(base_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path / f"log_reg_cv_results_{timestamp}.json"


def _write_run_results(run_results: Dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(_json_safe(run_results), file_handle, indent=2)
    print(f"Saved run results to {output_path}")
    return output_path


def verify_prepared_dataset_rows(
    prepared_data: PreparedDataset,
    db_manager: DatabaseManager,
    enabled: bool = False,
    sample_count: int = 5,
    random_state: int = 42,
    atol: float = 1e-8,
) -> Dict:
    """
    Verify sampled matrix rows against imported aggregate feature values.

    For each sampled row, this checks that the value in each configured
    (raw_feature_type, descriptive_stat_type) column matches what is imported
    from the corresponding aggregate feature entry.
    """
    if not enabled:
        return {"verification_run": False}

    n_rows = prepared_data.features.shape[0]
    if n_rows == 0:
        return {
            "verification_run": True,
            "sample_count_requested": sample_count,
            "sample_count_used": 0,
            "sampled_row_indices": [],
            "mismatch_count": 0,
            "is_pass": True,
            "details": [],
        }

    sample_size = min(sample_count, n_rows)
    rng = np.random.default_rng(random_state)
    sampled_indices = sorted(rng.choice(n_rows, size=sample_size, replace=False).tolist())

    mismatch_count = 0
    label_mismatch_count = 0
    details: List[Dict] = []
    for row_index in sampled_indices:
        observed_row = prepared_data.features[row_index]
        aggregate_feature_id = prepared_data.aggregate_feature_ids[row_index]
        user_id = prepared_data.user_ids[row_index]
        aggregate_entry = db_manager.import_data(
            [AggregateFeatureIdentifier(aggregate_feature_id)]
        )[0]
        user_data = db_manager.import_data([UserIdentifier(user_id)])[0]
        expected_label_bool = user_data.clinical_demographic_data.faller_status.to_bool()
        expected_label = (
            None if expected_label_bool is None else int(expected_label_bool)
        )
        observed_label = int(prepared_data.labels[row_index])
        label_match = (
            expected_label is not None and observed_label == expected_label
        )
        if not label_match:
            label_mismatch_count += 1

        row_mismatches: List[Dict] = []
        for feature_index, (raw_type, stat_type) in enumerate(prepared_data.feature_names):
            aggregate_feature = aggregate_entry.get_feature_from_type(raw_type)
            if aggregate_feature is None:
                expected = np.nan
            else:
                stat = aggregate_feature.get_statistic_from_type(stat_type)
                expected = np.nan if stat is None or stat.value is None else float(stat.value)
                if not np.isnan(expected) and np.isnan(float(expected)):
                    expected = np.nan

            observed = float(observed_row[feature_index])
            if np.isnan(expected) and np.isnan(observed):
                continue

            if np.isnan(expected) != np.isnan(observed) or not np.isclose(
                expected,
                observed,
                atol=atol,
                rtol=0.0,
            ):
                mismatch_count += 1
                row_mismatches.append(
                    {
                        "feature_index": feature_index,
                        "feature_key": f"{raw_type.value}::{stat_type.value}",
                        "observed": None if np.isnan(observed) else observed,
                        "expected": None if np.isnan(expected) else expected,
                    }
                )

        details.append(
            {
                "row_index": row_index,
                "user_id": user_id,
                "aggregate_feature_id": aggregate_feature_id,
                "observed_label": observed_label,
                "expected_label": expected_label,
                "label_match": label_match,
                "mismatch_count": len(row_mismatches),
                "mismatches": row_mismatches,
            }
        )

    return {
        "verification_run": True,
        "sample_count_requested": sample_count,
        "sample_count_used": sample_size,
        "sampled_row_indices": sampled_indices,
        "mismatch_count": mismatch_count,
        "label_mismatch_count": label_mismatch_count,
        "is_pass": mismatch_count == 0 and label_mismatch_count == 0,
        "details": details,
    }


class LogRegCVClassifier:

    def __init__(
        self,
        feature_id_file_path: str,
        db_manager: DatabaseManager,
        feature_config_json_path: Optional[str] = None,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.model_template = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.1, 0.5, 0.7, 0.9],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            max_iter=10000,
            scoring="roc_auc",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        self.pipeline_template = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", self.model_template),
            ]
        )
        self.db_manager: DatabaseManager = db_manager
        self.feature_id_file_path: str = feature_id_file_path
        self.feature_config_json_path: Optional[str] = feature_config_json_path
        self.feature_ids: List[AggregateFeatureIdentifier] = []
        self.selected_feature_pairs: Optional[
            List[tuple[RawFeatureType, DescriptiveStatisticType]]
        ] = None
        self.prepared_data: Optional[PreparedDataset] = None
        self.final_model_pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self._setup()

    def prepare_dataset(self) -> PreparedDataset:
        ft_preprocessor = FeaturePreprocessor(
            self.feature_ids,
            self.db_manager,
            selected_feature_pairs=self.selected_feature_pairs,
        )
        self.prepared_data = ft_preprocessor.preprocess()
        return self.prepared_data

    def train_model(self) -> None:
        """Fit final model on full dataset (for later inference use)."""
        if self.prepared_data is None:
            self.prepare_dataset()
        if self.prepared_data is None:
            raise RuntimeError("Unable to prepare training data.")

        self.final_model_pipeline = clone(self.pipeline_template)
        self.final_model_pipeline.fit(
            self.prepared_data.features,
            self.prepared_data.labels,
        )
        self.is_trained = True

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["sensitivity"] = float(recall_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["specificity"] = (
            float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        )
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
        metrics["tp"] = float(tp)
        metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))

        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        else:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
        return metrics

    @staticmethod
    def _aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not fold_metrics:
            raise ValueError("No fold metrics were produced.")

        metric_names = sorted(fold_metrics[0].keys())
        summary: Dict[str, float] = {"n_folds": float(len(fold_metrics))}
        for metric_name in metric_names:
            values = np.array([metrics[metric_name] for metrics in fold_metrics], dtype=float)
            summary[f"{metric_name}_mean"] = float(np.nanmean(values))
            summary[f"{metric_name}_std"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            )
        return summary

    def test_model(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        cv_random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate model via RepeatedStratifiedKFold and return aggregated metrics."""
        if self.prepared_data is None:
            self.prepare_dataset()
        if self.prepared_data is None:
            raise RuntimeError("Unable to prepare dataset for CV evaluation.")

        if cv_random_state is None:
            cv_random_state = self.random_state

        X = self.prepared_data.features
        y = self.prepared_data.labels
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=cv_random_state,
        )
        fold_metrics: List[Dict[str, float]] = []
        for train_index, test_index in splitter.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_pipeline = clone(self.pipeline_template)
            fold_pipeline.fit(X_train, y_train)
            y_pred = fold_pipeline.predict(X_test)
            y_proba = fold_pipeline.predict_proba(X_test)[:, 1]
            fold_metrics.append(self._compute_metrics(y_test, y_pred, y_proba))

        summary = self._aggregate_fold_metrics(fold_metrics)
        summary["n_splits"] = float(n_splits)
        summary["n_repeats"] = float(n_repeats)
        return summary

    def get_best_hyperparameters(self) -> Dict[str, float]:
        if not self.is_trained or self.final_model_pipeline is None:
            raise RuntimeError("Model must be trained before reading hyperparameters.")
        fitted_model: LogisticRegressionCV = self.final_model_pipeline.named_steps["clf"]
        best_c = float(np.ravel(fitted_model.C_)[0])
        best_l1_ratio = float(np.ravel(fitted_model.l1_ratio_)[0])
        return {"best_C": best_c, "best_l1_ratio": best_l1_ratio}

    def _setup(self):
        feature_path = Path(self.feature_id_file_path)
        with feature_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        # Accept either ["agg_id_1", ...] or {"feature_ids": ["agg_id_1", ...]}.
        if isinstance(data, list):
            id_values = data
        elif isinstance(data, dict) and isinstance(data.get("feature_ids"), list):
            id_values = data["feature_ids"]
        else:
            raise ValueError(
                "Feature ID file must be a list of IDs or an object with 'feature_ids'."
            )

        if not id_values:
            raise ValueError("No aggregate feature IDs found in feature ID file.")
        self.feature_ids = [AggregateFeatureIdentifier(str(value)) for value in id_values]
        if self.feature_config_json_path:
            self.selected_feature_pairs = _load_feature_pair_config_json(
                Path(self.feature_config_json_path)
            )


def _get_database_manager(base_path: Path) -> DatabaseManager:
    """Build database manager from converted data directory structure."""
    registry_paths = {
        IMUDataIdentifier: base_path / "registries" / "imu_data",
        UserIdentifier: base_path / "registries" / "user_data",
        RawFeatureIdentifier: base_path / "registries" / "raw_feature",
        AggregateFeatureIdentifier: base_path / "registries" / "agg_feature",
    }
    mapping_paths = {
        (IMUDataIdentifier, UserIdentifier): base_path
        / "mappings"
        / "imu_to_user_mapping",
        (RawFeatureIdentifier, IMUDataIdentifier): base_path
        / "mappings"
        / "raw_feat_to_imu_mapping",
        (AggregateFeatureIdentifier, RawFeatureIdentifier): base_path
        / "mappings"
        / "agg_feat_to_raw_feat_mapping",
    }
    output_dir_paths = {
        RawFeatureIdentifier: base_path / "raw_feat",
        AggregateFeatureIdentifier: base_path / "agg_feat",
    }
    return DatabaseGenerator().generate_database(
        registry_paths, mapping_paths, output_dir_paths, validate=False
    )


def generate_aggregate_feature_id_json(
    db_manager: DatabaseManager,
    output_path: Path,
) -> Path:
    """Export all aggregate feature IDs in the registry to a JSON file."""
    agg_registry = db_manager.registry_manager.get_provider(AggregateFeatureIdentifier)
    aggregate_feature_ids = sorted(list(agg_registry.registry.keys()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"feature_ids": aggregate_feature_ids}
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    print(f"Saved aggregate feature IDs to {output_path}")
    return output_path


def _resolve_raw_feature_type(value: str) -> RawFeatureType:
    for raw_type in RawFeatureType:
        if raw_type.value == value:
            return raw_type
    raise ValueError(f"Unknown raw feature type value in config: {value}")


def _resolve_descriptive_stat_type(value: str) -> DescriptiveStatisticType:
    for stat_type in DescriptiveStatisticType:
        if stat_type.value == value:
            return stat_type
    raise ValueError(f"Unknown descriptive stat type value in config: {value}")


def _load_feature_pair_config_json(
    config_path: Path,
) -> List[tuple[RawFeatureType, DescriptiveStatisticType]]:
    """
    Load configured feature pairs from JSON:
    {
      "feature_pairs": [
        {"raw_feature_type": "...", "descriptive_stat_type": "..."}
      ]
    }
    """
    with config_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    feature_pairs_payload = payload.get("feature_pairs")
    if not isinstance(feature_pairs_payload, list):
        raise ValueError("Feature config JSON must contain list field 'feature_pairs'.")

    feature_pairs: List[tuple[RawFeatureType, DescriptiveStatisticType]] = []
    for index, item in enumerate(feature_pairs_payload):
        if not isinstance(item, dict):
            raise ValueError(f"feature_pairs[{index}] must be an object.")
        raw_value = item.get("raw_feature_type")
        stat_value = item.get("descriptive_stat_type")
        if not isinstance(raw_value, str) or not isinstance(stat_value, str):
            raise ValueError(
                f"feature_pairs[{index}] requires string fields "
                "'raw_feature_type' and 'descriptive_stat_type'."
            )
        feature_pairs.append(
            (
                _resolve_raw_feature_type(raw_value),
                _resolve_descriptive_stat_type(stat_value),
            )
        )

    if not feature_pairs:
        raise ValueError("Feature config JSON contains zero feature_pairs.")
    return feature_pairs


def run_logistic_regression_cv(
    base_path: Path,
    feature_id_output_path: Optional[Path] = None,
    results_output_path: Optional[Path] = None,
    feature_config_json_path: Optional[Path] = None,
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 10,
    verify_training_matrix: bool = False,
    verification_sample_count: int = 5,
) -> Dict[str, float]:
    """
    Run logistic regression with elastic-net CV for converted-data dataset.

    Steps:
    1) Build database manager from base path
    2) Export aggregate feature IDs to JSON (same location by default)
    3) Evaluate using repeated CV and fit final model on full data
    4) Optionally verify sampled training-matrix rows
    5) Persist metrics + feature definitions to results JSON
    """
    db_manager = _get_database_manager(base_path)
    if feature_id_output_path is None:
        feature_id_output_path = base_path / "agg_feature_ids.json"
    if results_output_path is None:
        results_output_path = _default_results_output_path(base_path)

    feature_id_file = feature_id_output_path

    classifier = LogRegCVClassifier(
        feature_id_file_path=str(feature_id_file),
        db_manager=db_manager,
        feature_config_json_path=(
            str(feature_config_json_path) if feature_config_json_path else None
        ),
        random_state=random_state,
    )
    prepared_data = classifier.prepare_dataset()
    verification_summary = verify_prepared_dataset_rows(
        prepared_data=prepared_data,
        db_manager=db_manager,
        enabled=verify_training_matrix,
        sample_count=verification_sample_count,
        random_state=random_state,
    )
    if verify_training_matrix:
        print("Training-matrix verification:")
        print(
            f"  pass: {verification_summary['is_pass']}, "
            f"mismatches: {verification_summary['mismatch_count']}, "
            f"sampled_rows: {verification_summary['sample_count_used']}"
        )

    metrics = classifier.test_model(
        n_splits=n_splits,
        n_repeats=n_repeats,
        cv_random_state=random_state,
    )
    classifier.train_model()
    best_hyperparams = classifier.get_best_hyperparameters()

    print("Repeated CV summary metrics (mean +/- std):")
    for metric_name in sorted(
        [key[:-5] for key in metrics.keys() if key.endswith("_mean")]
    ):
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        if mean_key in metrics and std_key in metrics:
            print(f"  {metric_name}: {metrics[mean_key]:.4f} +/- {metrics[std_key]:.4f}")
    print(f"  n_folds: {int(metrics['n_folds'])}")
    print(f"  n_splits: {int(metrics['n_splits'])}")
    print(f"  n_repeats: {int(metrics['n_repeats'])}")
    print("Best hyperparameters:")
    print(f"  C: {best_hyperparams['best_C']:.6f}")
    print(f"  l1_ratio: {best_hyperparams['best_l1_ratio']:.4f}")

    class_counts = {
        "n_samples": int(len(prepared_data.labels)),
        "n_fallers": int(np.sum(prepared_data.labels == 1)),
        "n_non_fallers": int(np.sum(prepared_data.labels == 0)),
    }
    feature_definitions = _build_feature_definition_records(prepared_data.feature_names)
    run_results: Dict = {
        "run_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "base_path": str(base_path),
        "feature_id_file_path": str(feature_id_file),
        "feature_config_json_path": (
            str(feature_config_json_path) if feature_config_json_path else None
        ),
        "results_output_path": str(results_output_path),
        "cv_config": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "random_state": random_state,
        },
        "verification_config": {
            "verify_training_matrix": verify_training_matrix,
            "verification_sample_count": verification_sample_count,
        },
        "verification_summary": verification_summary,
        "class_counts": class_counts,
        "metrics": metrics,
        "best_hyperparameters": best_hyperparams,
        "feature_count": len(feature_definitions),
        "feature_definitions": feature_definitions,
    }
    _write_run_results(run_results, results_output_path)

    return {**run_results["metrics"], **run_results["best_hyperparameters"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run logistic regression CV classifier on aggregate gait features."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base path to converted data (e.g. .../ltmm_2026_02_21)",
    )
    parser.add_argument(
        "--feature-id-output",
        type=Path,
        default=None,
        help="Path for generated aggregate feature ID JSON (default: <base-path>/agg_feature_ids.json)",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=None,
        help="Path for persisted run results JSON (default: <base-path>/log_reg_cv_results_<timestamp>.json)",
    )
    parser.add_argument(
        "--feature-config-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file defining selected feature pairs to include. "
            "Expected schema: {'feature_pairs': [{'raw_feature_type': '...', "
            "'descriptive_stat_type': '...'}]}"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV and model reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds per CV repeat (default: 5)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeated CV runs (default: 10)",
    )
    parser.add_argument(
        "--verify-training-matrix",
        action="store_true",
        help="Run sampled row-level verification against imported aggregate features.",
    )
    parser.add_argument(
        "--verification-sample-count",
        type=int,
        default=5,
        help="Number of matrix rows to sample for verification (default: 5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_logistic_regression_cv(
        base_path=args.base_path,
        feature_id_output_path=args.feature_id_output,
        results_output_path=args.results_output,
        feature_config_json_path=args.feature_config_json,
        random_state=args.random_state,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        verify_training_matrix=args.verify_training_matrix,
        verification_sample_count=args.verification_sample_count,
    )