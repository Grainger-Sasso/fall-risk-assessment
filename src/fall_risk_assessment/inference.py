"""Assess a single user's fall risk from an IMU record with a trained bundle.

The assessor reproduces the training-time feature path exactly:
  IMU record -> GaitExtractionPipeline (bundle's backend/profile/epoch config)
  -> per-sample stride/epoch matrices -> column alignment onto the bundle's
  feature schema -> participant aggregation (nanmean) -> calibrated probability
  -> risk category.

Inputs can come either from loose record directories (imu_data.h5 +
user_data.json/clinical json + instrument spec) or from a SQLite-indexed
database via ``assess_db_record``.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_io.import_export.importers.data.imu.imu_data_importer import (
    IMUDataImporter,
)
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataImporter,
)
from src.data_io.import_export.importers.instrument_specifications.instrument_specification_importer import (
    InstrumentSpecificationImporter,
)
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.bout_sample_layout import (
    flatten_bout_features,
    usable_sample_mask,
)
from src.data_model.features.record_features import RecordFeatures
from src.data_model.instrument_specifications.imu_specifications import (
    IMUSpecifications,
)
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_types.feature.stride_feature_name import (
    StrideFeatureName,
    stride_feature_name_value,
)
from src.database_manager.database_manager import DatabaseManager
from src.fall_risk_assessment.model_bundle import FallRiskModelBundle
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.pipeline.gait_extraction_pipeline import GaitExtractionPipeline
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class BasisQuality:
    """Data-quality facts for one sampling basis of the assessed record."""

    n_usable_samples: int
    n_features_expected: int
    n_features_observed: int
    missing_features: List[str] = field(default_factory=list)
    nan_fraction: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_usable_samples": self.n_usable_samples,
            "n_features_expected": self.n_features_expected,
            "n_features_observed": self.n_features_observed,
            "missing_features": list(self.missing_features),
            "nan_fraction": round(float(self.nan_fraction), 4),
        }


@dataclass
class FallRiskAssessment:
    """The finished product: one user's fall-risk result with provenance."""

    user_id: str
    imu_record_id: str
    risk_probability: float
    risk_category: str
    predicted_faller: bool
    decision_threshold: float
    model_name: str
    assessed_at_utc: str
    feature_values: Dict[str, Optional[float]]
    feature_percentiles: Dict[str, Optional[float]]
    stride_quality: BasisQuality = None
    epoch_quality: BasisQuality = None
    known_faller_status: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "user_id": self.user_id,
            "imu_record_id": self.imu_record_id,
            "risk_probability": round(float(self.risk_probability), 4),
            "risk_category": self.risk_category,
            "predicted_faller": bool(self.predicted_faller),
            "decision_threshold": round(float(self.decision_threshold), 4),
            "model_name": self.model_name,
            "assessed_at_utc": self.assessed_at_utc,
            "known_faller_status": self.known_faller_status,
            "stride_quality": self.stride_quality.to_dict()
            if self.stride_quality
            else None,
            "epoch_quality": self.epoch_quality.to_dict()
            if self.epoch_quality
            else None,
            "feature_values": {
                name: (round(value, 6) if value is not None else None)
                for name, value in self.feature_values.items()
            },
            "feature_percentiles": {
                name: (round(value, 1) if value is not None else None)
                for name, value in self.feature_percentiles.items()
            },
        }

    def write_json(self, path: Path) -> Path:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


class FallRiskAssessor:
    """Runs the end-to-end assessment for single IMU records."""

    def __init__(self, bundle: FallRiskModelBundle):
        self.bundle = bundle
        config = bundle.extraction_config
        self.pipeline = GaitExtractionPipeline.create(
            backend=GaitExtractionBackendId(config.backend),
            profile=ExtractionProfile(config.profile),
            epoch_window_seconds=config.epoch_window_seconds,
            epoch_overlap_seconds=config.epoch_overlap_seconds,
        )

    @classmethod
    def from_bundle_dir(cls, bundle_dir: Path) -> "FallRiskAssessor":
        return cls(FallRiskModelBundle.load(Path(bundle_dir)))

    # ------------------------------------------------------------------ #
    # entry points
    # ------------------------------------------------------------------ #

    def assess_record_dirs(
        self,
        imu_dir: Path,
        user_dir: Path,
        instrument_spec_dir: Optional[Path] = None,
    ) -> FallRiskAssessment:
        imu_data = IMUDataImporter().import_data(Path(imu_dir))
        user_data = UserDataImporter().import_data(Path(user_dir))
        spec: Optional[IMUSpecifications] = None
        if instrument_spec_dir is not None:
            spec = InstrumentSpecificationImporter().import_data(
                Path(instrument_spec_dir)
            )
        return self.assess(imu_data, user_data, spec)

    def assess_db_record(
        self,
        db_manager: DatabaseManager,
        imu_id: IMUDataIdentifier,
    ) -> FallRiskAssessment:
        imu_data = db_manager.load_imu(imu_id)
        user_id = db_manager.get_user_for_imu(imu_id)
        if user_id is None:
            raise ValueError(f"No user mapping found for IMU '{imu_id.value}'.")
        user_data = db_manager.load_user(user_id)
        spec = None
        spec_id = db_manager.get_instrument_spec_for_imu(imu_id)
        if spec_id is not None:
            spec = db_manager.load_instrument_spec(spec_id)
        return self.assess(imu_data, user_data, spec)

    def assess(
        self,
        imu_data: IMUData,
        user_data: UserData,
        instrument_specifications: Optional[IMUSpecifications] = None,
    ) -> FallRiskAssessment:
        record_features = self.pipeline.run(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=instrument_specifications,
        )
        return self.assess_features(record_features, user_data)

    def assess_features(
        self,
        record_features: RecordFeatures,
        user_data: Optional[UserData] = None,
    ) -> FallRiskAssessment:
        """Assess from already-extracted RecordFeatures (no re-extraction)."""
        stride_vector, stride_quality = self._vectorize_basis(
            record_features, SampleBasis.STRIDE, self.bundle.stride_feature_names
        )
        epoch_vector, epoch_quality = self._vectorize_basis(
            record_features, SampleBasis.EPOCH, self.bundle.epoch_feature_names
        )
        feature_vector = np.concatenate([stride_vector, epoch_vector])

        probability = self.bundle.predict_probability(feature_vector)
        thresholds = self.bundle.thresholds

        feature_values = self._feature_value_map(feature_vector)
        known_status = None
        user_id = record_features.feature_metadata.user_identifier.value
        if user_data is not None:
            known_status = user_data.clinical_demographic_data.faller_status.value
            user_id = user_data.get_data_id().value

        return FallRiskAssessment(
            user_id=user_id,
            imu_record_id=record_features.feature_metadata.imu_data_identifier.value,
            risk_probability=probability,
            risk_category=thresholds.categorize(probability),
            predicted_faller=probability >= thresholds.decision_threshold,
            decision_threshold=thresholds.decision_threshold,
            model_name=self.bundle.model_name,
            assessed_at_utc=datetime.now(timezone.utc).isoformat(),
            feature_values=feature_values,
            feature_percentiles=self._feature_percentiles(feature_values),
            stride_quality=stride_quality,
            epoch_quality=epoch_quality,
            known_faller_status=known_status,
        )

    # ------------------------------------------------------------------ #
    # feature vector assembly (mirrors ClassificationDatasetBuilder)
    # ------------------------------------------------------------------ #

    def _vectorize_basis(
        self,
        record_features: RecordFeatures,
        basis: SampleBasis,
        expected_names: List[StrideFeatureName],
    ) -> Tuple[np.ndarray, BasisQuality]:
        bout = record_features.get_features_by_basis(basis)
        record_names = list(bout.feature_names)
        features = np.asarray(bout.features, dtype=float)

        if features.size == 0:
            quality = BasisQuality(
                n_usable_samples=0,
                n_features_expected=len(expected_names),
                n_features_observed=0,
                missing_features=[
                    stride_feature_name_value(name) for name in expected_names
                ],
                nan_fraction=1.0,
            )
            return np.full(len(expected_names), np.nan), quality

        per_sample = flatten_bout_features(features)
        per_sample = per_sample[usable_sample_mask(per_sample)]

        index_by_name = {name: idx for idx, name in enumerate(record_names)}
        aligned = np.full((per_sample.shape[0], len(expected_names)), np.nan)
        missing: List[str] = []
        for column, name in enumerate(expected_names):
            source = index_by_name.get(name)
            if source is None:
                missing.append(stride_feature_name_value(name))
                continue
            aligned[:, column] = per_sample[:, source]

        with np.errstate(all="ignore"):
            vector = (
                np.nanmean(aligned, axis=0)
                if aligned.shape[0] > 0
                else np.full(len(expected_names), np.nan)
            )
        nan_fraction = (
            float(np.mean(np.isnan(aligned))) if aligned.size > 0 else 1.0
        )
        quality = BasisQuality(
            n_usable_samples=int(aligned.shape[0]),
            n_features_expected=len(expected_names),
            n_features_observed=len(expected_names) - len(missing),
            missing_features=missing,
            nan_fraction=nan_fraction,
        )
        return vector, quality

    def _feature_value_map(
        self, feature_vector: np.ndarray
    ) -> Dict[str, Optional[float]]:
        values: Dict[str, Optional[float]] = {}
        for name, value in zip(self.bundle.feature_names, feature_vector):
            key = stride_feature_name_value(name)
            values[key] = float(value) if np.isfinite(value) else None
        return values

    def _feature_percentiles(
        self, feature_values: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        """Approximate percentile of each value within the training cohort."""
        percentiles: Dict[str, Optional[float]] = {}
        stats = self.bundle.cohort_feature_stats
        for name, value in feature_values.items():
            entry = stats.get(name)
            if value is None or not entry or entry.get("n", 0) < 2:
                percentiles[name] = None
                continue
            quantile_points = sorted(
                (float(key[1:]) / 100.0, float(val))
                for key, val in entry.items()
                if key.startswith("q")
            )
            if not quantile_points:
                percentiles[name] = None
                continue
            quantiles = [point[0] for point in quantile_points]
            cutoffs = [point[1] for point in quantile_points]
            # Clamp to the outermost stored quantiles rather than claiming
            # 0th/100th percentile for out-of-cohort values.
            percentiles[name] = float(
                np.interp(value, cutoffs, quantiles, left=quantiles[0], right=quantiles[-1])
                * 100.0
            )
        return percentiles
