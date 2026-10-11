import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.feature_validation.aggregation import aggregate_validation
from src.feature_validation.csv_export import (
    timestamped_path,
    write_integrity_csv,
    write_per_feature_csv,
    write_per_user_csv,
)
from src.feature_validation.feature_validator import (
    FeatureValidationCounts,
    RecordValidationResult,
)
from src.feature_validation.index_integrity import IndexIntegrityReport


def _counts(
    feature_type,
    usable=10,
    finite=10,
    implausible=0,
    atypical=0,
    unit="m/s",
    has_range=True,
    minimum=0.5,
    maximum=1.5,
    worst=None,
):
    return FeatureValidationCounts(
        feature_type=feature_type,
        unit=unit,
        usable_samples=usable,
        finite_count=finite,
        implausible_count=implausible,
        atypical_count=atypical,
        has_range=has_range,
        min_value=minimum,
        max_value=maximum,
        median_value=1.0,
        worst_implausible=worst,
    )


def _result(participant, counts, class_label="faller", feature_id="f1", usable=10):
    return RecordValidationResult(
        feature_id=feature_id,
        participant_id=participant,
        class_label=class_label,
        basis=SampleBasis.STRIDE,
        usable_samples=usable,
        counts=counts,
    )


class TestAggregation(unittest.TestCase):
    def test_empty_results_produce_empty_summary(self):
        summary = aggregate_validation([], SampleBasis.STRIDE)
        self.assertTrue(summary.is_empty)
        self.assertIn("No validated records", summary.summary_text())

    def test_per_user_totals_sum_across_features(self):
        results = [
            _result(
                "P1",
                [
                    _counts(FeatureType.GAIT_SPEED, finite=8, implausible=1, atypical=2),
                    _counts(FeatureType.CADENCE, finite=10, atypical=1),
                ],
            )
        ]
        summary = aggregate_validation(results, SampleBasis.STRIDE)
        user = summary.per_user[0]
        self.assertEqual("P1", user.participant_id)
        self.assertEqual(18, user.finite_values)
        self.assertEqual(2, user.missing_values)
        self.assertEqual(1, user.implausible_values)
        self.assertEqual(3, user.atypical_values)
        self.assertAlmostEqual(18 / 20, user.finite_fraction)
        self.assertAlmostEqual((18 - 1 - 3) / 18, user.meaningful_fraction)

    def test_per_user_records_are_counted_and_merged(self):
        results = [
            _result("P1", [_counts(FeatureType.GAIT_SPEED)], feature_id="f1"),
            _result("P1", [_counts(FeatureType.GAIT_SPEED)], feature_id="f2"),
            _result("P2", [_counts(FeatureType.GAIT_SPEED)], feature_id="f3"),
        ]
        summary = aggregate_validation(results, SampleBasis.STRIDE)
        by_id = {u.participant_id: u for u in summary.per_user}
        self.assertEqual(2, by_id["P1"].feature_records)
        self.assertEqual(1, by_id["P2"].feature_records)
        self.assertEqual(20, by_id["P1"].usable_samples)

    def test_fully_missing_feature_types_are_listed_per_user(self):
        results = [
            _result(
                "P1",
                [
                    _counts(FeatureType.GAIT_SPEED, finite=0, minimum=None, maximum=None),
                    _counts(FeatureType.CADENCE, finite=10),
                ],
            )
        ]
        summary = aggregate_validation(results, SampleBasis.STRIDE)
        self.assertEqual(["gait speed"], summary.per_user[0].feature_types_all_missing)

    def test_per_feature_aggregates_across_participants(self):
        results = [
            _result("P1", [_counts(FeatureType.GAIT_SPEED, implausible=1, worst=0.0, minimum=0.0)]),
            _result("P2", [_counts(FeatureType.GAIT_SPEED, minimum=0.8, maximum=2.0)]),
        ]
        summary = aggregate_validation(results, SampleBasis.STRIDE)
        feature = summary.per_feature[0]
        self.assertEqual(FeatureType.GAIT_SPEED, feature.feature_type)
        self.assertEqual(2, feature.participants)
        self.assertEqual(20, feature.finite_values)
        self.assertEqual(1, feature.implausible_values)
        self.assertEqual(0.0, feature.min_value)
        self.assertEqual(2.0, feature.max_value)
        self.assertEqual(["P1"], feature.participants_with_implausible)

    def test_per_feature_missing_and_finite_fractions_complement(self):
        results = [_result("P1", [_counts(FeatureType.GAIT_SPEED, usable=10, finite=6)])]
        feature = aggregate_validation(results, SampleBasis.STRIDE).per_feature[0]
        self.assertAlmostEqual(0.6, feature.finite_fraction)
        self.assertAlmostEqual(0.4, feature.missing_fraction)

    def test_summary_text_reports_offenders(self):
        results = [
            _result(
                "P1",
                [_counts(FeatureType.GAIT_SPEED, implausible=2, worst=99.0)],
            )
        ]
        text = aggregate_validation(results, SampleBasis.STRIDE).summary_text()
        self.assertIn("stride", text)
        self.assertIn("Most implausible", text)
        self.assertIn("gait speed", text)

    def test_zero_finite_values_do_not_divide_by_zero(self):
        results = [_result("P1", [_counts(FeatureType.GAIT_SPEED, usable=0, finite=0)])]
        summary = aggregate_validation(results, SampleBasis.STRIDE)
        self.assertEqual(0.0, summary.per_user[0].finite_fraction)
        self.assertEqual(0.0, summary.per_user[0].meaningful_fraction)
        self.assertEqual(0.0, summary.per_feature[0].implausible_fraction)


class TestCsvExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.summaries = [
            aggregate_validation(
                [
                    _result(
                        "P1",
                        [
                            _counts(
                                FeatureType.GAIT_SPEED, implausible=1, atypical=2, worst=0.0
                            ),
                            _counts(FeatureType.CADENCE, unit="steps/min"),
                        ],
                    )
                ],
                SampleBasis.STRIDE,
            )
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_timestamped_path_shape(self):
        generated = datetime(2026, 9, 16, 14, 30, 5)
        path = timestamped_path(self.dir / "validation.csv", "per_feature", generated)
        self.assertEqual("validation_per_feature_20260916_143005.csv", path.name)

    def test_per_feature_csv_roundtrip(self):
        path = write_per_feature_csv(self.dir / "features.csv", self.summaries)
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        self.assertEqual(2, len(rows))
        speed = next(r for r in rows if r["feature_type"] == "gait speed")
        self.assertEqual("stride", speed["basis"])
        self.assertEqual("1", speed["implausible_values"])
        self.assertEqual("2", speed["atypical_values"])
        self.assertEqual("m/s", speed["unit"])
        self.assertEqual("P1", speed["participants_with_implausible"])

    def test_per_user_csv_roundtrip(self):
        path = write_per_user_csv(self.dir / "users.csv", self.summaries)
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        self.assertEqual(1, len(rows))
        self.assertEqual("P1", rows[0]["participant_id"])
        self.assertEqual("faller", rows[0]["class_label"])
        self.assertEqual("1", rows[0]["implausible_values"])

    def test_integrity_csv_includes_drift_metrics(self):
        report = IndexIntegrityReport(
            indexed_feature_count=79,
            loadable_feature_count=41,
            dangling_feature_ids=["feature_a", "feature_b"],
            orphaned_payload_dirs=["/tmp/orphan"],
            indexed_imu_count=69,
            imu_ids_without_features=["imu_x"],
            participants_with_features={"P1"},
        )
        path = write_integrity_csv(self.dir / "integrity.csv", report)
        rows = {r["metric"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
        self.assertEqual("79", rows["indexed_feature_records"]["value"])
        self.assertEqual("41", rows["loadable_feature_records"]["value"])
        self.assertEqual("38", rows["index_overstatement"]["value"])
        self.assertEqual("2", rows["dangling_index_rows"]["value"])
        self.assertIn("feature_a", rows["dangling_index_rows"]["detail"])
        self.assertEqual("1", rows["orphaned_payload_dirs"]["value"])

    def test_export_creates_missing_parent_directories(self):
        nested = self.dir / "a" / "b" / "features.csv"
        path = write_per_feature_csv(nested, self.summaries)
        self.assertTrue(path.is_file())


class TestIndexIntegrityReport(unittest.TestCase):
    def test_clean_report(self):
        report = IndexIntegrityReport(
            indexed_feature_count=5, loadable_feature_count=5, indexed_imu_count=5
        )
        self.assertTrue(report.is_clean)
        self.assertEqual(0, report.index_overstatement)
        self.assertNotIn("WARNING", report.summary_text())

    def test_drift_is_flagged_in_summary(self):
        report = IndexIntegrityReport(
            indexed_feature_count=79,
            loadable_feature_count=41,
            dangling_feature_ids=["a"] * 38,
        )
        self.assertFalse(report.is_clean)
        self.assertEqual(38, report.index_overstatement)
        self.assertIn("WARNING", report.summary_text())
        self.assertIn("overstates", report.summary_text())


if __name__ == "__main__":
    unittest.main()
