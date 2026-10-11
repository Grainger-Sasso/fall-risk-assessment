"""Real ``RecordFeatures`` extracted with SKDH from assessment-database participant fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.user.user_data import UserData
from src.data_model.features.record_features import RecordFeatures
from src.data_model.instrument_specifications.imu_specifications import IMUSpecifications
from src.gait_features.backends.skdh.backend import SkdhBackend
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.contracts.extraction_result import GaitExtractionResult
from src.gait_features.processing.record_feature_builder import RecordFeatureBuilder
from test.fixtures.assessment_database import (
    DATASETS,
    ParticipantFixture,
    assessment_database_errors,
    load_participant_bundle,
    participant_fixtures,
)

DEFAULT_WINDOW_SECONDS = 8.0
DEFAULT_OVERLAP_SECONDS = 2.0


@dataclass(frozen=True)
class ExtractedParticipant:
    """One participant's real inputs, SKDH extraction result and built ``RecordFeatures``."""

    fixture: ParticipantFixture
    profile: ExtractionProfile
    imu_data: IMUData
    user_data: UserData
    instrument_specifications: IMUSpecifications
    extraction: GaitExtractionResult
    record: RecordFeatures

    @property
    def participant_id(self) -> str:
        return self.fixture.participant_id

    @property
    def class_label(self) -> str:
        return self.user_data.clinical_demographic_data.faller_status.value


def skdh_available() -> bool:
    try:
        import skdh  # noqa: F401
    except ImportError:
        return False
    return True


def feature_extraction_errors(datasets: Tuple[str, ...] = DATASETS) -> List[str]:
    """Reasons real features cannot be extracted: missing data fixtures or missing skdh."""
    errors: List[str] = []
    if not skdh_available():
        errors.append("scikit-digital-health (skdh) is not installed")
    errors.extend(assessment_database_errors(datasets))
    return errors


def build_record_features(
    extraction_result: GaitExtractionResult,
    imu_data: IMUData,
    user_data: UserData,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> RecordFeatures:
    builder = RecordFeatureBuilder(
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
    )
    return builder.build_from_extraction(
        extraction_result=extraction_result,
        imu_data=imu_data,
        user_data=user_data,
    )


@lru_cache(maxsize=None)
def extract_participant(
    fixture: ParticipantFixture,
    profile: ExtractionProfile = ExtractionProfile.TREADMILL,
) -> ExtractedParticipant:
    """Extract (once per process) and return a participant's real features.

    The returned objects are shared by every caller in the test process; copy the
    record (e.g. ``copy.deepcopy``) before mutating it.
    """
    imu_data, user_data, spec = load_participant_bundle(fixture)
    extraction = SkdhBackend(profile=profile).extract(
        imu_data=imu_data,
        user_data=user_data,
        instrument_specifications=spec,
    )
    return ExtractedParticipant(
        fixture=fixture,
        profile=profile,
        imu_data=imu_data,
        user_data=user_data,
        instrument_specifications=spec,
        extraction=extraction,
        record=build_record_features(extraction, imu_data, user_data),
    )


def extracted_participants(
    dataset: str = "lab_walks",
    tier: Optional[str] = None,
    profile: ExtractionProfile = ExtractionProfile.TREADMILL,
) -> List[ExtractedParticipant]:
    return [extract_participant(item, profile) for item in participant_fixtures(dataset, tier=tier)]
