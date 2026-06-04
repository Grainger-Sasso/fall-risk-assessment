import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.reporting.feature_report_generator import (
    _resolve_feature_types,
    _timestamped_path,
    _write_summary_csv,
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

    def test_write_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "report_summary.csv"
            _write_summary_csv(
                output,
                [
                    {
                        "feature_type": "gait speed",
                        "basis": "epoch",
                        "class_label": "faller",
                        "count": "10",
                        "mean": "1.2",
                        "median": "1.1",
                        "std": "0.3",
                        "iqr": "0.2",
                    }
                ],
            )
            contents = output.read_text(encoding="utf-8")
            self.assertIn("feature_type,basis,class_label,count,mean,median,std,iqr", contents)
            self.assertIn("gait speed,epoch,faller,10,1.2,1.1,0.3,0.2", contents)


if __name__ == "__main__":
    unittest.main()
