"""Serializable artifact describing a classification evaluation run.

This is the contract shared by the report generator and the visualization
plugin. JSON is written with NaN converted to null (valid JSON); metric values
are restored to NaN on load.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np


def _to_json_safe(value):
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _restore_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    restored: Dict[str, float] = {}
    for key, value in metrics.items():
        restored[key] = float("nan") if value is None else float(value)
    return restored


@dataclass
class ModelFamilyResult:
    """Results for one model family across bases and fusion strategies."""

    model_name: str
    base_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fusion_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    roc_curves: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    pr_curves: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    confusion: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_params: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass
class EvaluationArtifact:
    generated_at: str
    cv_config: Dict[str, object]
    participant_counts: Dict[str, int]
    sample_counts: Dict[str, int]
    fusion_strategies: List[str]
    model_results: List[ModelFamilyResult]
    ranking: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return _to_json_safe(asdict(self))

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "EvaluationArtifact":
        model_results: List[ModelFamilyResult] = []
        for entry in payload.get("model_results", []):
            model_results.append(
                ModelFamilyResult(
                    model_name=entry["model_name"],
                    base_results={
                        basis: _restore_metrics(metrics)
                        for basis, metrics in entry.get("base_results", {}).items()
                    },
                    fusion_results={
                        fusion: _restore_metrics(metrics)
                        for fusion, metrics in entry.get("fusion_results", {}).items()
                    },
                    roc_curves=entry.get("roc_curves", {}),
                    pr_curves=entry.get("pr_curves", {}),
                    confusion={
                        fusion: _restore_metrics(counts)
                        for fusion, counts in entry.get("confusion", {}).items()
                    },
                    best_params=entry.get("best_params", {}),
                )
            )
        return cls(
            generated_at=payload["generated_at"],
            cv_config=payload.get("cv_config", {}),
            participant_counts=payload.get("participant_counts", {}),
            sample_counts=payload.get("sample_counts", {}),
            fusion_strategies=payload.get("fusion_strategies", []),
            model_results=model_results,
            ranking=payload.get("ranking", []),
        )

    @classmethod
    def from_json(cls, path: Path) -> "EvaluationArtifact":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls.from_dict(payload)
