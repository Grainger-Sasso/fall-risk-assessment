import unittest

from src.gait_features.backends.skdh.backend import SkdhBackend
from src.gait_features.config.extraction_profile import ExtractionProfile
from test.fixtures.integration_cases import LtmmIntegrationTestCase
from test.fixtures.assessment_database import (
    load_participant_bundle,
    participant_fixtures,
)
from test.gait_features.validation.extraction_validators import (
    assert_bout_segments_valid,
    assert_walking_bouts_detected,
)


class TestSkdhBoutDetectionLtmm(LtmmIntegrationTestCase):
    """Whole-file LTMM extraction: non-faller (CO*) records contain walking bouts."""

    def test_primary_co_participants_detect_walking_bouts(self):
        for fixture in participant_fixtures("ltmm", tier="primary"):
            with self.subTest(participant=fixture.participant_id, imu_id=fixture.imu_id):
                imu_data, user_data, spec = load_participant_bundle(fixture)
                backend = SkdhBackend(profile=ExtractionProfile.FREE_LIVING)
                extraction = backend.extract(
                    imu_data=imu_data,
                    user_data=user_data,
                    instrument_specifications=spec,
                )
                assert_bout_segments_valid(
                    extraction.bout_segments,
                    min_bout_count=1,
                )
                assert_walking_bouts_detected(extraction.bout_segments)

    def test_secondary_co_participant_detects_walking_bouts(self):
        fixtures = participant_fixtures("ltmm", tier="secondary")
        self.assertGreaterEqual(len(fixtures), 1)
        fixture = fixtures[0]
        imu_data, user_data, spec = load_participant_bundle(fixture)
        backend = SkdhBackend(profile=ExtractionProfile.FREE_LIVING)
        extraction = backend.extract(
            imu_data=imu_data,
            user_data=user_data,
            instrument_specifications=spec,
        )
        assert_walking_bouts_detected(extraction.bout_segments)


if __name__ == "__main__":
    unittest.main()
