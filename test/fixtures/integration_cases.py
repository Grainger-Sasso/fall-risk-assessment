"""Base test cases for tests that run on real assessment-database fixtures.

These fail (rather than skip) when fixtures are unavailable so missing data is never
mistaken for a passing run. The LTMM whole-file case is the one opt-in exception.
"""

from __future__ import annotations

import unittest
from typing import List, Tuple

from test.fixtures.assessment_database import (
    DATASETS,
    ENV_RUN_LTMM,
    assessment_database_errors,
    ltmm_integration_requested,
)
from test.fixtures.real_features import feature_extraction_errors


def _raise_if_unavailable(errors: List[str], what: str) -> None:
    if errors:
        raise AssertionError(
            f"{what} are unavailable:\n" + "\n".join(f"  - {item}" for item in errors)
        )


class AssessmentDataTestCase(unittest.TestCase):
    """Requires the assessment database (indexes, IMU/user records) for ``fixture_datasets``."""

    fixture_datasets: Tuple[str, ...] = DATASETS

    @classmethod
    def setUpClass(cls) -> None:
        _raise_if_unavailable(
            assessment_database_errors(cls.fixture_datasets),
            "Assessment database fixtures",
        )


class SkdhIntegrationTestCase(AssessmentDataTestCase):
    """Requires the assessment database and skdh, so real features can be extracted."""

    @classmethod
    def setUpClass(cls) -> None:
        _raise_if_unavailable(
            feature_extraction_errors(cls.fixture_datasets),
            "Real feature extraction fixtures",
        )


class LtmmIntegrationTestCase(SkdhIntegrationTestCase):
    """Whole-file LTMM tests; skipped unless the LTMM opt-in env var is set."""

    @classmethod
    def setUpClass(cls) -> None:
        if not ltmm_integration_requested():
            raise unittest.SkipTest(
                f"Set {ENV_RUN_LTMM}=1 to run LTMM whole-file integration tests."
            )
        super().setUpClass()
