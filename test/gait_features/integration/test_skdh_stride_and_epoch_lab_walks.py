import unittest

from test.fixtures.assessment_database import participant_fixtures
from test.fixtures.integration_cases import SkdhIntegrationTestCase
from test.fixtures.real_features import extract_participant
from test.gait_features.validation.extraction_validators import (
    assert_bout_segments_valid,
    assert_epoch_features_reliable,
    assert_epoch_window_geometry,
    assert_record_features_schema,
    assert_stride_features_reliable,
    assert_walking_bouts_detected,
)
from test.gait_features.validation.physiological_validators import (
    assert_epoch_physiological_plausibility,
    assert_stride_physiological_plausibility,
)
from test.gait_features.validation.reference_validators import (
    assert_gait_speed_matches_reference,
)


class TestSkdhStrideAndEpochLabWalks(SkdhIntegrationTestCase):
    """Treadmill Lab Walks records validate stride and epoch extraction."""

    def test_primary_lab_walks_records_produce_stride_and_epoch_features(self):
        for fixture in participant_fixtures("lab_walks", tier="primary"):
            with self.subTest(participant=fixture.participant_id, imu_id=fixture.imu_id):
                extracted = extract_participant(fixture)
                extraction = extracted.extraction
                assert_bout_segments_valid(extraction.bout_segments, min_bout_count=1)
                assert_walking_bouts_detected(extraction.bout_segments)

                record = extracted.record
                assert_record_features_schema(record)
                assert_epoch_window_geometry(record)
                assert_epoch_features_reliable(record)
                assert_stride_features_reliable(extraction, record)
                assert_stride_physiological_plausibility(record)
                assert_epoch_physiological_plausibility(record)
                assert_gait_speed_matches_reference(
                    record,
                    fixture.participant_id,
                )
                self.assertGreater(record.stride_features.usable_sample_count, 5)


if __name__ == "__main__":
    unittest.main()
