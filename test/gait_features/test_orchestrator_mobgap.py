import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.contracts.extraction_result import GaitExtractionResult
from src.gait_features.contracts.provenance import ExtractionProvenance
from src.gait_features.gait_feature_dataset_orchestrator import GaitFeatureDatasetOrchestrator
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class TestOrchestratorMobgap(unittest.TestCase):
    def test_mobgap_backend_wiring(self):
        db_manager = MagicMock()
        db_manager.list_imu_ids.return_value = [IMUDataIdentifier("imu_1")]
        db_manager.get_user_for_imu.return_value = UserIdentifier("user_1")
        db_manager.load_imu.return_value = "imu_data"
        db_manager.load_user.return_value = "user_data"
        db_manager.get_instrument_spec_for_imu.return_value = None

        extractor = MagicMock()
        extraction_result = GaitExtractionResult(
            provenance=ExtractionProvenance.create(
                backend="mobgap",
                stride_feature_catalog="mobgap",
                profile=ExtractionProfile.MOBGAP_HEALTHY.value,
                library_version="1.2.0",
            ),
            bout_segments=[],
            stride_feature_names=[],
            stride_feature_series={},
        )
        extractor.extract.return_value = extraction_result
        builder = MagicMock()
        builder.build_from_extraction.return_value = SimpleNamespace(
            feature_metadata=SimpleNamespace(feature_identifier=SimpleNamespace(value="feature_mobgap"))
        )

        orchestrator = GaitFeatureDatasetOrchestrator(
            db_manager=db_manager,
            extraction_backend=GaitExtractionBackendId.MOBGAP,
            extraction_profile=ExtractionProfile.MOBGAP_HEALTHY,
            extractor=extractor,
            record_feature_builder=builder,
        )
        orchestrator._write_last_run_summary = MagicMock()
        summary = orchestrator.generate_for_all_imu()

        extractor.extract.assert_called_once()
        builder.build_from_extraction.assert_called_once()
        self.assertEqual(summary.extraction_backend, "mobgap")
        self.assertEqual(summary.extraction_profile, "mobgap_healthy")
        self.assertEqual([item.value for item in summary.generated_feature_ids], ["feature_mobgap"])


if __name__ == "__main__":
    unittest.main()
