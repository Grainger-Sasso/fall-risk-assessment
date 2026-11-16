"""Train and export a deployable fall-risk model from a feature database.

Flow:
  1. Build the participant-level early-fusion dataset (stride + epoch feature
     blocks aggregated per participant) from the SQLite-backed DatabaseManager.
  2. Optionally exclude held-out participants entirely (they can later be
     assessed through the inference pipeline as true unseen test users).
  3. Rank model families with the existing repeated, participant-stratified
     cross-validation evaluator and pick the champion.
  4. Derive calibrated out-of-fold probabilities for the champion to choose the
     binary decision threshold (max balanced accuracy).
  5. Fit the final tuned + calibrated champion on all training participants and
     export a ``FallRiskModelBundle``.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score

from src.classification.data.classification_dataset import (
    EarlyFusionParticipantDataset,
)
from src.classification.data.classification_dataset_builder import (
    ClassificationDatasetBuilder,
)
from src.classification.evaluation.cross_validation import (
    InsufficientDataError,
    iter_participant_folds,
    resolve_n_splits,
)
from src.classification.evaluation.evaluation_artifact import EvaluationArtifact
from src.classification.evaluation.participant_early_fusion_evaluator import (
    ParticipantEarlyFusionEvaluator,
)
from src.classification.models.model_factory import available_model_names, create_model
from src.database_manager.database_generator import DatabaseGenerator
from src.data_types.feature.stride_feature_name import stride_feature_name_value
from src.fall_risk_assessment.model_bundle import (
    ExtractionConfig,
    FallRiskModelBundle,
    RiskThresholds,
)

COHORT_STAT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class TrainingResult:
    bundle: FallRiskModelBundle
    bundle_dir: Path
    evaluation_artifact: EvaluationArtifact
    evaluation_artifact_path: Path
    holdout_participant_ids: List[str] = field(default_factory=list)


@dataclass
class FallRiskModelTrainer:
    """Trains the champion fall-risk model and exports it as a bundle."""

    sqlite_db_path: Path
    output_dir: Path
    extraction_config: ExtractionConfig = field(default_factory=ExtractionConfig)
    model_names: Optional[List[str]] = None
    holdout_participant_ids: List[str] = field(default_factory=list)
    aggregation: str = "mean"
    n_splits: int = 5
    n_repeats: int = 5
    inner_splits: int = 3
    random_state: int = 42
    fast_mode: bool = False
    calibration_method: str = "sigmoid"
    verbose: bool = True

    def train(self) -> TrainingResult:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        full_dataset, sample_counts = self._build_dataset()
        train_dataset, excluded = self._exclude_holdouts(full_dataset)
        self._log(
            f"dataset: {train_dataset.n_participants} training participants "
            f"({int(np.sum(train_dataset.y == 1))} fallers, "
            f"{int(np.sum(train_dataset.y == 0))} non-fallers), "
            f"{len(excluded)} held out, {train_dataset.n_features} features"
        )

        artifact = self._rank_model_families(train_dataset, sample_counts)
        champion_name = self._champion_model_name(artifact)
        self._log(f"champion model family: {champion_name}")

        oof_y, oof_p = self._champion_out_of_fold_probabilities(
            train_dataset, champion_name
        )
        thresholds = self._derive_thresholds(oof_y, oof_p)
        self._log(
            f"decision threshold (max balanced accuracy on OOF): "
            f"{thresholds.decision_threshold:.3f}"
        )

        final_model = self._fit_final_model(train_dataset, champion_name)

        generated_at = datetime.now(timezone.utc)
        timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
        artifact_path = self.output_dir / f"model_selection_cv_{timestamp}.json"
        artifact.to_json(artifact_path)

        bundle = FallRiskModelBundle(
            model=final_model,
            model_name=champion_name,
            stride_feature_names=list(train_dataset.stride_feature_names),
            epoch_feature_names=list(train_dataset.epoch_feature_names),
            extraction_config=self.extraction_config,
            thresholds=thresholds,
            aggregation=self.aggregation,
            calibration_method=self.calibration_method,
            training_provenance={
                "trained_at_utc": generated_at.isoformat(),
                "sqlite_db_path": str(Path(self.sqlite_db_path).resolve()),
                "n_training_participants": train_dataset.n_participants,
                "n_fallers": int(np.sum(train_dataset.y == 1)),
                "n_non_fallers": int(np.sum(train_dataset.y == 0)),
                "training_participant_ids": list(train_dataset.participant_ids),
                "holdout_participant_ids": excluded,
                "random_state": self.random_state,
                "model_selection_artifact": artifact_path.name,
            },
            cv_metrics=self._champion_cv_metrics(artifact, champion_name),
            cohort_feature_stats=self._cohort_feature_stats(train_dataset),
        )
        bundle_dir = self.output_dir / f"fall_risk_model_{champion_name}_{timestamp}"
        bundle.save(bundle_dir)
        self._log(f"bundle exported: {bundle_dir}")

        return TrainingResult(
            bundle=bundle,
            bundle_dir=bundle_dir,
            evaluation_artifact=artifact,
            evaluation_artifact_path=artifact_path,
            holdout_participant_ids=excluded,
        )

    # ------------------------------------------------------------------ #
    # dataset assembly
    # ------------------------------------------------------------------ #

    def _build_dataset(
        self,
    ) -> Tuple[EarlyFusionParticipantDataset, Dict[str, int]]:
        db_manager = DatabaseGenerator().generate_database(
            Path(self.sqlite_db_path), validate=True
        )
        builder = ClassificationDatasetBuilder(db_manager=db_manager)
        sample_dataset = builder.build(
            stride_catalog=self.extraction_config.stride_catalog
        )
        participant_dataset = builder.build_early_fusion_participant_dataset(
            aggregation=self.aggregation,
            stride_catalog=self.extraction_config.stride_catalog,
        )
        if participant_dataset.is_empty:
            raise InsufficientDataError(
                "No participant feature rows available; run feature extraction "
                "before training."
            )
        sample_counts = {
            "stride": sample_dataset.stride.n_samples,
            "epoch": sample_dataset.epoch.n_samples,
        }
        return participant_dataset, sample_counts

    def _exclude_holdouts(
        self, dataset: EarlyFusionParticipantDataset
    ) -> Tuple[EarlyFusionParticipantDataset, List[str]]:
        holdout = {pid for pid in self.holdout_participant_ids}
        if not holdout:
            return dataset, []
        keep_mask = np.array(
            [pid not in holdout for pid in dataset.participant_ids], dtype=bool
        )
        excluded = [pid for pid in dataset.participant_ids if pid in holdout]
        missing = sorted(holdout - set(dataset.participant_ids))
        if missing:
            self._log(
                f"warning: holdout participants not present in dataset: {missing}"
            )
        filtered = EarlyFusionParticipantDataset(
            X=dataset.X[keep_mask],
            y=dataset.y[keep_mask],
            participant_ids=[
                pid for pid, keep in zip(dataset.participant_ids, keep_mask) if keep
            ],
            stride_feature_names=list(dataset.stride_feature_names),
            epoch_feature_names=list(dataset.epoch_feature_names),
            aggregation=dataset.aggregation,
        )
        return filtered, excluded

    # ------------------------------------------------------------------ #
    # model selection + final fit
    # ------------------------------------------------------------------ #

    def _rank_model_families(
        self,
        dataset: EarlyFusionParticipantDataset,
        sample_counts: Dict[str, int],
    ) -> EvaluationArtifact:
        evaluator = ParticipantEarlyFusionEvaluator(
            dataset=dataset,
            sample_counts=sample_counts,
            model_names=self.model_names or available_model_names(),
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            inner_splits=self.inner_splits,
            random_state=self.random_state,
            fast_mode=self.fast_mode,
            verbose=self.verbose,
        )
        return evaluator.evaluate()

    @staticmethod
    def _champion_model_name(artifact: EvaluationArtifact) -> str:
        if not artifact.ranking:
            raise InsufficientDataError("Model evaluation produced no ranking.")
        return str(artifact.ranking[0]["model_name"])

    @staticmethod
    def _champion_cv_metrics(
        artifact: EvaluationArtifact, champion_name: str
    ) -> Dict[str, object]:
        for result in artifact.model_results:
            if result.model_name == champion_name:
                return {
                    "fusion_results": result.fusion_results,
                    "ranking": [
                        entry
                        for entry in artifact.ranking
                        if entry["model_name"] == champion_name
                    ],
                }
        return {}

    def _champion_out_of_fold_probabilities(
        self,
        dataset: EarlyFusionParticipantDataset,
        champion_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(dataset.y, dtype=int)
        n_splits = resolve_n_splits(labels, desired=self.n_splits)
        oof_y: List[int] = []
        oof_p: List[float] = []
        for train_idx, test_idx in iter_participant_folds(
            labels, n_splits, self.n_repeats, self.random_state
        ):
            model = self._fit_calibrated(
                dataset.X[train_idx], labels[train_idx], champion_name
            )
            probs = model.predict_proba(dataset.X[test_idx])[:, 1]
            oof_y.extend(labels[test_idx].tolist())
            oof_p.extend(probs.tolist())
        return np.asarray(oof_y, dtype=int), np.asarray(oof_p, dtype=float)

    def _fit_final_model(
        self, dataset: EarlyFusionParticipantDataset, champion_name: str
    ):
        return self._fit_calibrated(
            dataset.X, np.asarray(dataset.y, dtype=int), champion_name
        )

    def _fit_calibrated(self, X: np.ndarray, y: np.ndarray, model_name: str):
        """Tune the family's pipeline on (X, y), then CV-calibrate it."""
        model = create_model(
            model_name,
            random_state=self.random_state,
            fast_mode=self.fast_mode,
            enable_tuning=True,
        )
        model.fit(X, y, groups=None)
        tuned_pipeline = model._fitted_estimator
        calibration_splits = resolve_n_splits(
            y, desired=min(5, self.n_splits), minimum=2
        )
        calibrated = CalibratedClassifierCV(
            estimator=clone(tuned_pipeline),
            method=self.calibration_method,
            cv=calibration_splits,
        )
        calibrated.fit(X, y)
        return calibrated

    @staticmethod
    def _derive_thresholds(oof_y: np.ndarray, oof_p: np.ndarray) -> RiskThresholds:
        """Pick the binary operating point maximizing balanced accuracy on OOF."""
        finite = np.isfinite(oof_p)
        oof_y = oof_y[finite]
        oof_p = oof_p[finite]
        if oof_y.size == 0 or len(np.unique(oof_y)) < 2:
            return RiskThresholds()
        candidates = np.unique(np.round(oof_p, 4))
        best_threshold, best_score = 0.5, -1.0
        for threshold in candidates:
            score = balanced_accuracy_score(oof_y, (oof_p >= threshold).astype(int))
            if score > best_score:
                best_threshold, best_score = float(threshold), float(score)
        return RiskThresholds(decision_threshold=best_threshold)

    # ------------------------------------------------------------------ #
    # cohort context
    # ------------------------------------------------------------------ #

    def _cohort_feature_stats(
        self, dataset: EarlyFusionParticipantDataset
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        names = [
            stride_feature_name_value(name)
            for name in (
                list(dataset.stride_feature_names) + list(dataset.epoch_feature_names)
            )
        ]
        for column, name in enumerate(names):
            values = dataset.X[:, column]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                stats[name] = {"n": 0.0}
                continue
            entry = {
                "n": float(finite.size),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            }
            for quantile in COHORT_STAT_QUANTILES:
                entry[f"q{int(quantile * 100):02d}"] = float(
                    np.quantile(finite, quantile)
                )
            stats[name] = entry
        return stats

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[fall-risk-train] {message}")
