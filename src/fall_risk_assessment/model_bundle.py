"""Versioned, self-describing artifact for a trained fall-risk model.

A bundle directory contains:
  - ``model.joblib``: the fitted (and optionally calibrated) sklearn estimator
  - ``manifest.json``: everything needed to reproduce the feature vector a new
    record must be mapped onto (feature schema, aggregation, extraction
    config), plus decision thresholds, training provenance, CV metrics, and
    training-cohort feature statistics for report context.

The manifest, not the pickle, is the source of truth for schema compatibility:
``FallRiskModelBundle.load`` refuses to load bundles with an unknown schema
version so stale artifacts fail loudly instead of silently mis-predicting.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.data_types.feature.stride_feature_name import (
    StrideFeatureName,
    parse_stride_feature_name,
    stride_feature_name_value,
)

BUNDLE_SCHEMA_VERSION = "1.0"
MODEL_FILE_NAME = "model.joblib"
MANIFEST_FILE_NAME = "manifest.json"

RISK_CATEGORY_LOW = "low"
RISK_CATEGORY_MODERATE = "moderate"
RISK_CATEGORY_HIGH = "high"


@dataclass
class RiskThresholds:
    """Probability cutoffs that turn a calibrated risk probability into decisions.

    ``decision_threshold`` is the binary faller/non-faller operating point
    (chosen on out-of-fold predictions during training). ``low_boundary`` and
    ``high_boundary`` band the probability into low/moderate/high categories.
    """

    decision_threshold: float = 0.5
    low_boundary: float = 1.0 / 3.0
    high_boundary: float = 2.0 / 3.0

    def categorize(self, probability: float) -> str:
        if probability < self.low_boundary:
            return RISK_CATEGORY_LOW
        if probability >= self.high_boundary:
            return RISK_CATEGORY_HIGH
        return RISK_CATEGORY_MODERATE

    def to_dict(self) -> Dict[str, float]:
        return {
            "decision_threshold": float(self.decision_threshold),
            "low_boundary": float(self.low_boundary),
            "high_boundary": float(self.high_boundary),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, float]) -> "RiskThresholds":
        return cls(
            decision_threshold=float(payload["decision_threshold"]),
            low_boundary=float(payload["low_boundary"]),
            high_boundary=float(payload["high_boundary"]),
        )


@dataclass
class ExtractionConfig:
    """Feature-extraction settings a record must be processed with at inference."""

    backend: str = "skdh"
    profile: str = "free_living"
    epoch_window_seconds: float = 8.0
    epoch_overlap_seconds: float = 2.0
    stride_catalog: str = "skdh"

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "profile": self.profile,
            "epoch_window_seconds": float(self.epoch_window_seconds),
            "epoch_overlap_seconds": float(self.epoch_overlap_seconds),
            "stride_catalog": self.stride_catalog,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ExtractionConfig":
        return cls(
            backend=str(payload["backend"]),
            profile=str(payload["profile"]),
            epoch_window_seconds=float(payload["epoch_window_seconds"]),
            epoch_overlap_seconds=float(payload["epoch_overlap_seconds"]),
            stride_catalog=str(payload["stride_catalog"]),
        )


@dataclass
class FallRiskModelBundle:
    """A trained fall-risk classifier plus everything inference needs."""

    model: object
    model_name: str
    stride_feature_names: List[StrideFeatureName]
    epoch_feature_names: List[StrideFeatureName]
    extraction_config: ExtractionConfig
    thresholds: RiskThresholds
    aggregation: str = "mean"
    evaluation_mode: str = "early_fusion_participant"
    calibration_method: Optional[str] = "sigmoid"
    training_provenance: Dict[str, object] = field(default_factory=dict)
    cv_metrics: Dict[str, object] = field(default_factory=dict)
    cohort_feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    schema_version: str = BUNDLE_SCHEMA_VERSION

    @property
    def feature_names(self) -> List[StrideFeatureName]:
        """Full input-vector feature order: stride block then epoch block."""
        return list(self.stride_feature_names) + list(self.epoch_feature_names)

    @property
    def n_features(self) -> int:
        return len(self.stride_feature_names) + len(self.epoch_feature_names)

    def predict_probability(self, feature_vector: np.ndarray) -> float:
        """Calibrated faller probability for one participant feature vector."""
        row = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        if row.shape[1] != self.n_features:
            raise ValueError(
                f"Feature vector has {row.shape[1]} values; bundle expects "
                f"{self.n_features} ({len(self.stride_feature_names)} stride + "
                f"{len(self.epoch_feature_names)} epoch)."
            )
        return float(self.model.predict_proba(row)[:, 1][0])

    def save(self, bundle_dir: Path) -> Path:
        import json

        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, bundle_dir / MODEL_FILE_NAME)
        manifest = {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "evaluation_mode": self.evaluation_mode,
            "aggregation": self.aggregation,
            "calibration_method": self.calibration_method,
            "stride_feature_names": [
                stride_feature_name_value(name) for name in self.stride_feature_names
            ],
            "epoch_feature_names": [
                stride_feature_name_value(name) for name in self.epoch_feature_names
            ],
            "extraction_config": self.extraction_config.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "training_provenance": self.training_provenance,
            "cv_metrics": self.cv_metrics,
            "cohort_feature_stats": self.cohort_feature_stats,
        }
        with (bundle_dir / MANIFEST_FILE_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: Path) -> "FallRiskModelBundle":
        import json

        bundle_dir = Path(bundle_dir)
        manifest_path = bundle_dir / MANIFEST_FILE_NAME
        model_path = bundle_dir / MODEL_FILE_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Bundle model not found: {model_path}")

        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

        schema_version = manifest.get("schema_version")
        if schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported bundle schema version '{schema_version}'; this build "
                f"supports '{BUNDLE_SCHEMA_VERSION}'. Retrain or migrate the bundle."
            )

        return cls(
            model=joblib.load(model_path),
            model_name=manifest["model_name"],
            evaluation_mode=manifest["evaluation_mode"],
            aggregation=manifest["aggregation"],
            calibration_method=manifest.get("calibration_method"),
            stride_feature_names=[
                parse_stride_feature_name(value)
                for value in manifest["stride_feature_names"]
            ],
            epoch_feature_names=[
                parse_stride_feature_name(value)
                for value in manifest["epoch_feature_names"]
            ],
            extraction_config=ExtractionConfig.from_dict(manifest["extraction_config"]),
            thresholds=RiskThresholds.from_dict(manifest["thresholds"]),
            training_provenance=manifest.get("training_provenance", {}),
            cv_metrics=manifest.get("cv_metrics", {}),
            cohort_feature_stats=manifest.get("cohort_feature_stats", {}),
            schema_version=schema_version,
        )
