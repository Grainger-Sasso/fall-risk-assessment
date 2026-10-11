"""Feature validator behaviour on real SKDH features from Lab Walks participants.

Complements the synthetic tests: here the inputs are the tensors the pipeline really
produces (NaN padding, structural missingness, genuine outliers), and every validator
count is checked against an independent numpy recount rather than trusted.
"""

import copy
import csv
import dataclasses
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_model.features.bout_features import BoutFeatures
from src.data_types.feature.feature_type import FeatureType
from src.data_types.sample_basis.sample_basis import SampleBasis
from src.data_visualization.suite.services.visualization_data_service import VisualizationDataService
from src.feature_validation.aggregation import aggregate_validation
from src.feature_validation.csv_export import write_per_feature_csv
from src.feature_validation.feature_validator import (
    FeatureValidationCounts,
    RecordValidationResult,
    validate_record,
)
from src.feature_validation.index_integrity import audit_index_integrity
from src.feature_validation.range_catalog import (
    JUDGMENT,
    LITERATURE_DIRECT,
    Bounds,
    FeatureRange,
    RangeCatalog,
    default_range_catalog,
)
from src.feature_validation.validation_service import ValidationRun, run_feature_validation
from src.gait_features.config.extraction_backend import GaitExtractionBackendId
from src.gait_features.config.extraction_profile import ExtractionProfile
from src.gait_features.gait_feature_dataset_orchestrator import GaitFeatureDatasetOrchestrator
from test.fixtures.assessment_database import (
    TempDatabaseContext,
    imu_identifier,
    participant_fixtures,
    temp_database_manager,
)
from test.fixtures.integration_cases import SkdhIntegrationTestCase
from test.fixtures.real_features import ExtractedParticipant, extract_participant, extracted_participants

BASES: Tuple[SampleBasis, ...] = (SampleBasis.STRIDE, SampleBasis.EPOCH)
MAX_IMPLAUSIBLE_FRACTION = 0.005


def _limits(bounds: Optional[Bounds]) -> Tuple[float, float]:
    if bounds is None:
        return -np.inf, np.inf
    low = -np.inf if bounds.minimum is None else bounds.minimum
    high = np.inf if bounds.maximum is None else bounds.maximum
    return low, high


def _usable_rows(bout: BoutFeatures) -> np.ndarray:
    """(usable_sample, feature) matrix built directly from the (bout, feature, sample) tensor."""
    tensor = np.asarray(bout.features, dtype=float)
    rows = np.moveaxis(tensor, 1, -1).reshape(-1, tensor.shape[1])
    return rows[~np.all(np.isnan(rows), axis=1)]


def _recount(values: np.ndarray, feature_range: FeatureRange) -> Tuple[int, int, int]:
    """Return (finite, implausible, atypical) using vectorised, inclusive bounds."""
    finite = values[np.isfinite(values)]
    outer_low, outer_high = _limits(feature_range.implausible)
    implausible = (finite < outer_low) | (finite > outer_high)
    if feature_range.atypical is None:
        atypical = np.zeros_like(implausible)
    else:
        inner_low, inner_high = _limits(feature_range.atypical)
        atypical = ~implausible & ((finite < inner_low) | (finite > inner_high))
    return int(finite.size), int(implausible.sum()), int(atypical.sum())


def _stack_bouts(longer: BoutFeatures, shorter: BoutFeatures) -> BoutFeatures:
    """Two real single-bout tensors as one two-bout tensor; the shorter is NaN-padded."""
    pad = longer.num_samples - shorter.num_samples
    padded = np.pad(
        shorter.features, ((0, 0), (0, 0), (0, pad)), constant_values=np.nan
    )
    second_start = float(longer.bout_ends[-1]) + 1.0
    second_end = second_start + float(shorter.bout_ends[0] - shorter.bout_starts[0])
    return dataclasses.replace(
        longer,
        features=np.concatenate([longer.features, padded], axis=0),
        bout_starts=np.array([longer.bout_starts[0], second_start]),
        bout_ends=np.array([longer.bout_ends[0], second_end]),
    )


def _counts_for(result: RecordValidationResult, feature_type: FeatureType) -> FeatureValidationCounts:
    return next(item for item in result.counts if item.feature_type == feature_type)


def _validate(
    participant: ExtractedParticipant, basis: SampleBasis, catalog: RangeCatalog
) -> RecordValidationResult:
    return validate_record(
        record=participant.record,
        basis=basis,
        catalog=catalog,
        class_label=participant.class_label,
    )


class TestFeatureValidatorOnRealLabWalksFeatures(SkdhIntegrationTestCase):
    """Validator counts on every Lab Walks fixture participant (controls and a faller)."""

    fixture_datasets = ("lab_walks",)
    participants: List[ExtractedParticipant]
    catalog: RangeCatalog
    results: Dict[Tuple[str, SampleBasis], RecordValidationResult]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog = default_range_catalog()
        cls.participants = extracted_participants("lab_walks")
        cls.results = {
            (participant.participant_id, basis): _validate(participant, basis, cls.catalog)
            for participant in cls.participants
            for basis in BASES
        }

    def _cases(self) -> List[Tuple[ExtractedParticipant, SampleBasis, RecordValidationResult]]:
        return [
            (participant, basis, self.results[(participant.participant_id, basis)])
            for participant in self.participants
            for basis in BASES
        ]

    def test_fixtures_span_both_cohorts(self) -> None:
        labels = {participant.class_label for participant in self.participants}
        self.assertIn("faller", labels)
        self.assertIn("non-faller", labels)

    def test_every_extracted_feature_type_has_a_range(self) -> None:
        for participant, basis, result in self._cases():
            with self.subTest(participant=participant.participant_id, basis=basis.value):
                self.assertFalse(result.is_empty)
                self.assertEqual(result.uncovered_feature_types, [])
                self.assertTrue(all(item.has_range for item in result.counts))
                expected = participant.record.get_features_by_basis(basis).feature_names
                self.assertEqual([item.feature_type for item in result.counts], list(expected))

    def test_usable_samples_exclude_padding(self) -> None:
        for participant, basis, result in self._cases():
            with self.subTest(participant=participant.participant_id, basis=basis.value):
                bout = participant.record.get_features_by_basis(basis)
                self.assertEqual(result.usable_samples, bout.usable_sample_count)
                self.assertEqual(result.usable_samples, _usable_rows(bout).shape[0])
                self.assertEqual(result.participant_id, participant.participant_id)
                self.assertEqual(result.class_label, participant.class_label)

    def test_padding_from_multi_bout_records_is_not_counted(self) -> None:
        """Lab Walks records are single-bout, so padding is exercised by stacking two."""
        for basis in BASES:
            with self.subTest(basis=basis.value):
                by_length = sorted(
                    self.participants,
                    key=lambda p: p.record.get_features_by_basis(basis).num_samples,
                )
                shorter, longer = by_length[0], by_length[-1]
                longer_bout = longer.record.get_features_by_basis(basis)
                shorter_bout = shorter.record.get_features_by_basis(basis)
                self.assertGreater(longer_bout.num_samples, shorter_bout.num_samples)

                record = copy.deepcopy(longer.record)
                stacked = _stack_bouts(longer_bout, shorter_bout)
                if basis == SampleBasis.STRIDE:
                    record.stride_features = stacked
                else:
                    record.epoch_features = stacked
                result = validate_record(record, basis, self.catalog)

                parts = [self.results[(p.participant_id, basis)] for p in (longer, shorter)]
                self.assertEqual(result.usable_samples, sum(p.usable_samples for p in parts))
                self.assertLess(result.usable_samples, stacked.num_bouts * stacked.num_samples)
                for column, counts in enumerate(result.counts):
                    expected = [part.counts[column] for part in parts]
                    self.assertEqual(counts.finite_count, sum(c.finite_count for c in expected))
                    self.assertEqual(
                        counts.implausible_count, sum(c.implausible_count for c in expected)
                    )
                    self.assertEqual(counts.atypical_count, sum(c.atypical_count for c in expected))

    def test_counts_match_independent_recount(self) -> None:
        for participant, basis, result in self._cases():
            bout = participant.record.get_features_by_basis(basis)
            rows = _usable_rows(bout)
            for column, feature_type in enumerate(bout.feature_names):
                with self.subTest(
                    participant=participant.participant_id,
                    basis=basis.value,
                    feature=feature_type.value,
                ):
                    counts = result.counts[column]
                    feature_range = self.catalog.get(feature_type)
                    finite, implausible, atypical = _recount(rows[:, column], feature_range)
                    self.assertEqual(counts.usable_samples, rows.shape[0])
                    self.assertEqual(counts.finite_count, finite)
                    self.assertEqual(counts.implausible_count, implausible)
                    self.assertEqual(counts.atypical_count, atypical)

    def test_counts_are_internally_consistent(self) -> None:
        for participant, basis, result in self._cases():
            for counts in result.counts:
                with self.subTest(
                    participant=participant.participant_id,
                    basis=basis.value,
                    feature=counts.feature_type.value,
                ):
                    self.assertEqual(counts.finite_count + counts.missing_count, counts.usable_samples)
                    self.assertGreaterEqual(counts.in_range_count, 0)
                    if counts.finite_count:
                        self.assertLessEqual(counts.min_value, counts.median_value)
                        self.assertLessEqual(counts.median_value, counts.max_value)
                    feature_range = self.catalog.get(counts.feature_type)
                    if counts.implausible_count:
                        self.assertFalse(feature_range.implausible.contains(counts.worst_implausible))
                    else:
                        self.assertIsNone(counts.worst_implausible)
                        self.assertTrue(feature_range.implausible.contains(counts.min_value))
                        self.assertTrue(feature_range.implausible.contains(counts.max_value))

    def test_implausible_values_are_rare(self) -> None:
        for participant, basis, result in self._cases():
            with self.subTest(participant=participant.participant_id, basis=basis.value):
                self.assertLessEqual(
                    result.implausible_total,
                    MAX_IMPLAUSIBLE_FRACTION * result.finite_total,
                )

    def test_known_swing_time_outlier_is_flagged_implausible(self) -> None:
        """CO003 has one real stride with a 0.920 s swing, just above the 0.9 s ceiling."""
        result = self.results[("CO003", SampleBasis.STRIDE)]
        swing = _counts_for(result, FeatureType.SWING_TIME)
        self.assertEqual(swing.implausible_count, 1)
        self.assertAlmostEqual(swing.worst_implausible, 0.920, places=3)
        self.assertEqual(swing.implausible_provenance, JUDGMENT)

    def test_gait_speed_is_within_the_literature_normative_band(self) -> None:
        for participant in self.participants:
            with self.subTest(participant=participant.participant_id):
                result = self.results[(participant.participant_id, SampleBasis.STRIDE)]
                speed = _counts_for(result, FeatureType.GAIT_SPEED)
                feature_range = self.catalog.get(FeatureType.GAIT_SPEED)
                self.assertEqual(speed.atypical_provenance, LITERATURE_DIRECT)
                self.assertEqual(speed.implausible_count, 0)
                self.assertTrue(feature_range.atypical.contains(speed.median_value))

    def test_injected_defects_are_counted_exactly(self) -> None:
        source = next(p for p in self.participants if p.participant_id == "CO001")
        baseline = self.results[("CO001", SampleBasis.STRIDE)]
        record = copy.deepcopy(source.record)
        bout = record.stride_features
        column = list(bout.feature_names).index(FeatureType.GAIT_SPEED)
        feature_range = self.catalog.get(FeatureType.GAIT_SPEED)

        slice_ = bout.features[:, column, :]
        in_band = np.isfinite(slice_) & (slice_ >= feature_range.atypical.minimum) & (
            slice_ <= feature_range.atypical.maximum
        )
        positions = list(zip(*np.nonzero(in_band)))[:6]
        self.assertEqual(len(positions), 6, "need six in-band gait-speed values to perturb")

        implausible_value = feature_range.implausible.maximum + 2.0
        atypical_value = feature_range.atypical.minimum - 0.2
        for index, (bout_index, sample_index) in enumerate(positions):
            value = implausible_value if index < 3 else atypical_value if index < 5 else np.nan
            bout.features[bout_index, column, sample_index] = value

        result = validate_record(record, SampleBasis.STRIDE, self.catalog)
        before = _counts_for(baseline, FeatureType.GAIT_SPEED)
        after = _counts_for(result, FeatureType.GAIT_SPEED)

        self.assertEqual(result.usable_samples, baseline.usable_samples)
        self.assertEqual(after.implausible_count, before.implausible_count + 3)
        self.assertEqual(after.atypical_count, before.atypical_count + 2)
        self.assertEqual(after.finite_count, before.finite_count - 1)
        self.assertEqual(after.worst_implausible, implausible_value)
        for unchanged_before, unchanged_after in zip(baseline.counts, result.counts):
            if unchanged_before.feature_type == FeatureType.GAIT_SPEED:
                continue
            self.assertEqual(unchanged_before, unchanged_after)

        self.assertEqual(
            _counts_for(_validate(source, SampleBasis.STRIDE, self.catalog), FeatureType.GAIT_SPEED),
            before,
        )

    def test_aggregation_totals_equal_per_record_totals(self) -> None:
        for basis in BASES:
            with self.subTest(basis=basis.value):
                results = [self.results[(p.participant_id, basis)] for p in self.participants]
                summary = aggregate_validation(results, basis)

                by_user = {row.participant_id: row for row in summary.per_user}
                self.assertEqual(set(by_user), {p.participant_id for p in self.participants})
                for result in results:
                    row = by_user[result.participant_id]
                    self.assertEqual(row.class_label, result.class_label)
                    self.assertEqual(row.finite_values, result.finite_total)
                    self.assertEqual(row.implausible_values, result.implausible_total)
                    self.assertEqual(row.atypical_values, result.atypical_total)

                for row in summary.per_feature:
                    per_record = [_counts_for(result, row.feature_type) for result in results]
                    self.assertEqual(row.participants, len(results))
                    self.assertEqual(row.finite_values, sum(c.finite_count for c in per_record))
                    self.assertEqual(
                        row.implausible_values, sum(c.implausible_count for c in per_record)
                    )
                    self.assertEqual(row.atypical_values, sum(c.atypical_count for c in per_record))
                    self.assertEqual(
                        sorted(row.participants_with_implausible),
                        sorted(
                            result.participant_id
                            for result, c in zip(results, per_record)
                            if c.implausible_count
                        ),
                    )

    def test_per_feature_csv_reports_the_same_totals(self) -> None:
        summaries = [
            aggregate_validation(
                [self.results[(p.participant_id, basis)] for p in self.participants], basis
            )
            for basis in BASES
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_per_feature_csv(Path(tmp) / "per_feature.csv", summaries)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        for summary in summaries:
            with self.subTest(basis=summary.basis.value):
                csv_rows = {
                    row["feature_type"]: row for row in rows if row["basis"] == summary.basis.value
                }
                self.assertEqual(len(csv_rows), len(summary.per_feature))
                for feature in summary.per_feature:
                    row = csv_rows[feature.feature_type.value]
                    self.assertEqual(int(row["finite_values"]), feature.finite_values)
                    self.assertEqual(int(row["implausible_values"]), feature.implausible_values)
                    self.assertEqual(int(row["atypical_values"]), feature.atypical_values)
                    self.assertEqual(row["implausible_provenance"], feature.implausible_provenance)
                    self.assertEqual(row["atypical_provenance"], feature.atypical_provenance)


class TestFeatureValidationRunOnRealIndex(SkdhIntegrationTestCase):
    """End to end: orchestrator writes features to an isolated index copy, then the
    validation service audits that index and validates what it can load back."""

    fixture_datasets = ("lab_walks",)
    fixtures = participant_fixtures("lab_walks", tier="primary")
    context: TempDatabaseContext
    feature_root: Path
    validation_run: ValidationRun

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._stack = ExitStack()
        try:
            cls.context = cls._stack.enter_context(temp_database_manager("lab_walks"))
            cls.feature_root = cls.context.root / "features"
            orchestrator = GaitFeatureDatasetOrchestrator(
                db_manager=cls.context.db_manager,
                treadmill_profile=True,
                extraction_backend=GaitExtractionBackendId.SKDH,
                extraction_profile=ExtractionProfile.TREADMILL,
            )
            orchestrator._write_last_run_summary = lambda summary: None  # noqa: SLF001
            summary = orchestrator.generate_for_imu_ids(
                imu_ids=[imu_identifier(item.imu_id) for item in cls.fixtures]
            )
            if summary.status != "success":
                raise AssertionError(f"Feature generation failed: {summary}")
            cls.validation_run = run_feature_validation(
                VisualizationDataService(db_manager=cls.context.db_manager),
                feature_payload_roots=[cls.feature_root],
            )
        except BaseException:
            cls._stack.close()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls._stack.close()
        super().tearDownClass()

    def test_generated_features_are_fully_indexed_and_loadable(self) -> None:
        integrity = self.validation_run.integrity
        expected = len(self.fixtures)
        self.assertEqual(integrity.indexed_feature_count, expected)
        self.assertEqual(integrity.loadable_feature_count, expected)
        self.assertEqual(integrity.dangling_feature_ids, [])
        self.assertEqual(integrity.unreadable_feature_ids, {})
        self.assertEqual(integrity.orphaned_payload_dirs, [])
        self.assertEqual(
            set(integrity.participants_with_features),
            {item.participant_id for item in self.fixtures},
        )
        self.assertEqual(
            integrity.indexed_imu_count - len(integrity.imu_ids_without_features), expected
        )
        self.assertEqual(self.validation_run.load_warnings, [])

    def test_stray_payload_directory_is_reported_as_orphan(self) -> None:
        stray = self.feature_root / "features_not_in_index"
        stray.mkdir()
        try:
            report = audit_index_integrity(
                self.context.db_manager, feature_payload_roots=[self.feature_root]
            )
        finally:
            stray.rmdir()
        self.assertEqual(report.orphaned_payload_dirs, [str(stray)])
        self.assertEqual(report.loadable_feature_count, len(self.fixtures))

    def test_persisted_features_validate_like_in_memory_extraction(self) -> None:
        catalog = default_range_catalog()
        for basis in BASES:
            summary = self.validation_run.summary_for(basis)
            by_user = {row.participant_id: row for row in summary.per_user}
            for fixture in self.fixtures:
                with self.subTest(basis=basis.value, participant=fixture.participant_id):
                    participant = extract_participant(fixture)
                    expected = _validate(participant, basis, catalog)
                    row = by_user[fixture.participant_id]
                    self.assertEqual(row.class_label, participant.class_label)
                    self.assertEqual(row.usable_samples, expected.usable_samples)
                    self.assertEqual(row.finite_values, expected.finite_total)
                    self.assertEqual(row.implausible_values, expected.implausible_total)
                    self.assertEqual(row.atypical_values, expected.atypical_total)


if __name__ == "__main__":
    unittest.main()
