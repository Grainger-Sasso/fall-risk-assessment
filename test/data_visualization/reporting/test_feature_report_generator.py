import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.reporting.feature_report_generator import (
    _resolve_feature_types,
    _timestamped_path,
    _write_feature_level_csv,
    _write_participant_level_csv,
)


class TestFeatureReportGeneratorHelpers(unittest.TestCase):
    def test_resolve_feature_types_filters_by_value(self):
        available = [FeatureType.GAIT_SPEED, FeatureType.CADENCE]
        resolved = _resolve_feature_types("gait speed", available)
        self.assertEqual(resolved, [FeatureType.GAIT_SPEED])

    def test_timestamped_path_includes_basis_and_datetime(self):
        generated_at = datetime(2026, 6, 3, 9, 16, 0)
        base = Path("/tmp/reports/feature_report.pdf")

        pdf_path = _timestamped_path(base, SampleBasis.EPOCH, generated_at, ".pdf")
        csv_path = _timestamped_path(base, SampleBasis.EPOCH, generated_at, ".csv")

        self.assertEqual(pdf_path.name, "feature_report_epoch_20260603_091600.pdf")
        self.assertEqual(csv_path.name, "feature_report_epoch_20260603_091600.csv")
        self.assertEqual(pdf_path.parent, base.parent)
        self.assertEqual(csv_path.parent, base.parent)

    def test_write_feature_level_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "feature_summary.csv"
            _write_feature_level_csv(
                output,
                [
                    {
                        "report_level": "feature",
                        "feature_type": "gait speed",
                        "basis": "epoch",
                        "valid_count": "10",
                        "total_usable_samples": "12",
                        "valid_fraction": "0.833333",
                        "missing_fraction": "0.166667",
                        "cohens_d": "1.200000",
                    }
                ],
            )
            contents = output.read_text(encoding="utf-8")
            self.assertIn("report_level,feature_type,basis,valid_count", contents)
            self.assertIn("feature,gait speed,epoch,10,12", contents)

    def test_write_participant_level_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "participant_summary.csv"
            _write_participant_level_csv(
                output,
                [
                    {
                        "report_level": "participant",
                        "participant_id": "user_1",
                        "class_label": "faller",
                        "feature_id": "feature_1",
                        "basis": "stride",
                        "usable_sample_count": "80",
                        "tensor_slot_count": "100",
                        "padded_slot_count": "20",
                        "mean_coverage": "0.950000",
                        "count_consistent": "True",
                    }
                ],
            )
            contents = output.read_text(encoding="utf-8")
            self.assertIn("participant_id,class_label,feature_id", contents)
            self.assertIn("participant,user_1,faller,feature_1,stride,80,100,20", contents)


if __name__ == "__main__":
    unittest.main()
