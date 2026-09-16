import os
import unittest

from test.gait_features.fixtures.fixture_loader import (
    ltmm_integration_requested,
    require_integration_fixtures,
)


class GaitIntegrationTestCase(unittest.TestCase):
    """Integration tests that fail (not skip) when external fixtures are unavailable."""

    @classmethod
    def setUpClass(cls) -> None:
        require_integration_fixtures()


class LtmmIntegrationTestCase(GaitIntegrationTestCase):
    """Whole-file LTMM tests; skipped unless RUN_LTMM_INTEGRATION_TESTS is set."""

    @classmethod
    def setUpClass(cls) -> None:
        if not ltmm_integration_requested():
            raise unittest.SkipTest(
                "Set RUN_LTMM_INTEGRATION_TESTS=1 to run LTMM whole-file integration tests."
            )
        require_integration_fixtures()
