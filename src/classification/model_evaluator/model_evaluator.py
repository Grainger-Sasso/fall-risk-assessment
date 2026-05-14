import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np

from src.classification.feature_preprocessor.feature_preprocessor import (
    FeaturePreprocessor,
    PreparedDataset,
)
from src.classification.models.base_classifier_model import BaseClassifierModel
from src.classification.models.boosting.lightgbm_model import LightGbmShallowModel
from src.classification.models.log_reg.logistic_regression_model import (
    LogisticRegressionElasticNetModel,
)
from src.classification.models.svm.linear_svm_model import LinearSvmModel
from src.classification.models.svm.rbf_svm_model import RbfSvmModel
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


def _get_database_manager(base_path: Path) -> DatabaseManager:
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


class ModelEvaluator:
    def __init__(
        self,
        db_manager: DatabaseManager,
        aggregate_feature_ids: List[AggregateFeatureIdentifier],
        models: List[BaseClassifierModel],
        selected_feature_pairs: Optional[
            List[tuple[RawFeatureType, DescriptiveStatisticType]]
        ] = None,
    ):
        self.db_manager = db_manager
        self.aggregate_feature_ids = aggregate_feature_ids
        self.models = models
        self.selected_feature_pairs = selected_feature_pairs

    def _prepare_dataset(self) -> PreparedDataset:
        preprocessor = FeaturePreprocessor(
            feature_ids=self.aggregate_feature_ids,
            db_manager=self.db_manager,
            selected_feature_pairs=self.selected_feature_pairs,
        )
        return preprocessor.preprocess()

    @staticmethod
    def _rank_models(model_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
        def score(result: Dict[str, object]) -> float:
            metrics = result["metrics"]
            roc = metrics.get("roc_auc_mean", np.nan)
            pr = metrics.get("pr_auc_mean", np.nan)
            roc_value = -1.0 if roc is None or np.isnan(roc) else float(roc)
            pr_value = -1.0 if pr is None or np.isnan(pr) else float(pr)
            return 0.5 * roc_value + 0.5 * pr_value

        ordered = sorted(model_results, key=score, reverse=True)
        ranking: List[Dict[str, object]] = []
        for rank, row in enumerate(ordered, start=1):
            metrics = row["metrics"]
            ranking.append(
                {
                    "rank": rank,
                    "model_name": row["model_name"],
                    "roc_auc_mean": metrics.get("roc_auc_mean"),
                    "pr_auc_mean": metrics.get("pr_auc_mean"),
                    "score": score(row),
                }
            )
        return ranking

    def evaluate(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> Dict[str, object]:
        prepared_data = self._prepare_dataset()
        model_results: List[Dict[str, object]] = []
        for model in self.models:
            evaluation = model.evaluate(
                prepared_data=prepared_data,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            model_results.append(
                {
                    "model_name": evaluation.model_name,
                    "metrics": evaluation.metrics,
                    "best_hyperparameters": evaluation.best_hyperparameters,
                }
            )

        class_counts = {
            "n_samples": int(len(prepared_data.labels)),
            "n_fallers": int(np.sum(prepared_data.labels == 1)),
            "n_non_fallers": int(np.sum(prepared_data.labels == 0)),
        }
        report = {
            "run_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "cv_config": {
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "random_state": random_state,
            },
            "class_counts": class_counts,
            "feature_count": len(prepared_data.feature_names),
            "feature_definitions": _build_feature_definition_records(
                prepared_data.feature_names
            ),
            "model_results": model_results,
            "model_ranking": self._rank_models(model_results),
        }
        return report


def _load_aggregate_feature_ids(path: Path) -> List[AggregateFeatureIdentifier]:
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict) and isinstance(data.get("feature_ids"), list):
        values = data["feature_ids"]
    else:
        raise ValueError(
            "Feature ID file must be a list of IDs or an object with 'feature_ids'."
        )
    if not values:
        raise ValueError("No aggregate feature IDs found in feature ID file.")
    return [AggregateFeatureIdentifier(str(value)) for value in values]


def _default_results_output_path(output_dir: Path, fast_mode: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "fast" if fast_mode else "full"
    return output_dir / f"model_evaluator_report_{mode_suffix}_{timestamp}.json"


def _resolve_cv_config(n_splits: int, n_repeats: int, fast_mode: bool) -> Dict[str, int]:
    if not fast_mode:
        return {"n_splits": n_splits, "n_repeats": n_repeats}
    return {
        "n_splits": min(n_splits, 3),
        "n_repeats": min(n_repeats, 2),
    }


def run_model_evaluator(
    base_path: Path,
    feature_id_file: Path,
    results_output_dir: Optional[Path] = None,
    feature_config_json_path: Optional[Path] = None,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
    fast_mode: bool = False,
) -> Dict[str, object]:
    db_manager = _get_database_manager(base_path)
    aggregate_feature_ids = _load_aggregate_feature_ids(feature_id_file)
    selected_feature_pairs = (
        _load_feature_pair_config_json(feature_config_json_path)
        if feature_config_json_path
        else None
    )

    models: List[BaseClassifierModel] = [
        LogisticRegressionElasticNetModel(
            random_state=random_state, fast_mode=fast_mode
        ),
        LinearSvmModel(random_state=random_state, fast_mode=fast_mode),
        RbfSvmModel(random_state=random_state, fast_mode=fast_mode),
        LightGbmShallowModel(random_state=random_state, fast_mode=fast_mode),
    ]
    evaluator = ModelEvaluator(
        db_manager=db_manager,
        aggregate_feature_ids=aggregate_feature_ids,
        models=models,
        selected_feature_pairs=selected_feature_pairs,
    )
    effective_cv = _resolve_cv_config(
        n_splits=n_splits, n_repeats=n_repeats, fast_mode=fast_mode
    )
    print(
        "Model evaluator mode: "
        f"{'fast' if fast_mode else 'full'} "
        f"(n_splits={effective_cv['n_splits']}, n_repeats={effective_cv['n_repeats']})"
    )

    model_results: List[Dict[str, object]] = []
    print("Preparing dataset...")
    prepared_data = evaluator._prepare_dataset()
    print(f"Prepared dataset with {len(prepared_data.labels)} samples.")
    for model_index, model in enumerate(evaluator.models, start=1):
        start_time = perf_counter()
        print(
            f"[{model_index}/{len(evaluator.models)}] "
            f"Running {model.model_name}..."
        )
        try:
            evaluation = model.evaluate(
                prepared_data=prepared_data,
                n_splits=effective_cv["n_splits"],
                n_repeats=effective_cv["n_repeats"],
                random_state=random_state,
            )
        except Exception:
            elapsed_seconds = perf_counter() - start_time
            print(
                f"[{model_index}/{len(evaluator.models)}] "
                f"{model.model_name} failed after {elapsed_seconds:.2f}s"
            )
            raise
        elapsed_seconds = perf_counter() - start_time
        print(
            f"[{model_index}/{len(evaluator.models)}] "
            f"{model.model_name} completed in {elapsed_seconds:.2f}s"
        )
        model_results.append(
            {
                "model_name": evaluation.model_name,
                "metrics": evaluation.metrics,
                "best_hyperparameters": evaluation.best_hyperparameters,
            }
        )

    class_counts = {
        "n_samples": int(len(prepared_data.labels)),
        "n_fallers": int(np.sum(prepared_data.labels == 1)),
        "n_non_fallers": int(np.sum(prepared_data.labels == 0)),
    }
    report = {
        "run_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cv_config": {
            "n_splits_requested": n_splits,
            "n_repeats_requested": n_repeats,
            "n_splits_effective": effective_cv["n_splits"],
            "n_repeats_effective": effective_cv["n_repeats"],
            "random_state": random_state,
            "fast_mode": fast_mode,
        },
        "class_counts": class_counts,
        "feature_count": len(prepared_data.feature_names),
        "feature_definitions": _build_feature_definition_records(
            prepared_data.feature_names
        ),
        "model_results": model_results,
        "model_ranking": evaluator._rank_models(model_results),
    }
    report["base_path"] = str(base_path)
    report["feature_id_file"] = str(feature_id_file)
    report["feature_config_json_path"] = (
        str(feature_config_json_path) if feature_config_json_path else None
    )

    if results_output_dir is None:
        results_output_dir = base_path
    results_output_path = _default_results_output_path(
        output_dir=results_output_dir, fast_mode=fast_mode
    )
    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    with results_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(_json_safe(report), file_handle, indent=2)
    print(f"Saved model evaluator report to {results_output_path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standardized model evaluation across multiple classifiers."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base path to converted data (e.g. .../ltmm_2026_02_21)",
    )
    parser.add_argument(
        "--feature-id-file",
        type=Path,
        required=True,
        help="Path to aggregate feature ID JSON (list or {'feature_ids': []})",
    )
    parser.add_argument(
        "--results-output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for report JSON. File is auto-named as "
            "model_evaluator_report_<fast|full>_<timestamp>.json "
            "(default directory: <base-path>)."
        ),
    )
    parser.add_argument(
        "--feature-config-json",
        type=Path,
        default=None,
        help="Optional feature-pair filter JSON.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV/model reproducibility (default: 42)",
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
        "--fast-mode",
        action="store_true",
        help=(
            "Run a reduced-compute evaluation configuration. "
            "By default this is off and a full run is used."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_model_evaluator(
        base_path=args.base_path,
        feature_id_file=args.feature_id_file,
        results_output_dir=args.results_output_dir,
        feature_config_json_path=args.feature_config_json,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        fast_mode=args.fast_mode,
    )
