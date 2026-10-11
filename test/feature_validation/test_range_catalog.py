import json
import tempfile
import unittest
from pathlib import Path

from src.data_types.feature.feature_type import FeatureType
from src.feature_validation.range_catalog import (
    ATYPICAL,
    IMPLAUSIBLE,
    Bounds,
    RangeCatalogError,
    default_range_catalog,
    load_range_catalog,
)


def _write_catalog(payload: dict) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, handle)
    handle.close()
    return Path(handle.name)


class TestBounds(unittest.TestCase):
    def test_inclusive_bounds(self):
        bounds = Bounds(minimum=1.0, maximum=2.0)
        self.assertTrue(bounds.contains(1.0))
        self.assertTrue(bounds.contains(2.0))
        self.assertFalse(bounds.contains(0.999))
        self.assertFalse(bounds.contains(2.001))

    def test_none_bound_is_unbounded(self):
        self.assertTrue(Bounds(minimum=0.0).contains(1e9))
        self.assertTrue(Bounds(maximum=0.0).contains(-1e9))

    def test_nan_is_not_a_range_violation(self):
        # NaN is a missingness concern and must not be double-counted as out of range.
        self.assertTrue(Bounds(minimum=0.0, maximum=1.0).contains(float("nan")))


class TestBundledCatalog(unittest.TestCase):
    def setUp(self):
        self.catalog = default_range_catalog()

    def test_covers_every_stride_and_epoch_feature_type(self):
        for feature_types in (
            FeatureType.get_stride_feature_types(),
            FeatureType.get_epoch_feature_types(),
        ):
            self.assertEqual([], self.catalog.missing_coverage(feature_types))

    def test_every_entry_has_at_least_one_tier(self):
        for feature_range in self.catalog.ranges.values():
            self.assertTrue(
                feature_range.implausible is not None or feature_range.atypical is not None,
                "{} has no bounds".format(feature_range.feature_type),
            )

    def test_every_reference_key_resolves_to_a_citation(self):
        for feature_range in self.catalog.ranges.values():
            if feature_range.reference_key:
                self.assertTrue(
                    feature_range.citation,
                    "{} cites '{}' with no citation text".format(
                        feature_range.feature_type, feature_range.reference_key
                    ),
                )

    def test_atypical_never_wider_than_implausible(self):
        """An atypical band outside the implausible band could never be reached."""
        for feature_range in self.catalog.ranges.values():
            implausible = feature_range.implausible
            atypical = feature_range.atypical
            if implausible is None or atypical is None:
                continue
            if implausible.minimum is not None and atypical.minimum is not None:
                self.assertGreaterEqual(
                    atypical.minimum,
                    implausible.minimum,
                    "{}: atypical minimum is below implausible minimum".format(
                        feature_range.feature_type
                    ),
                )
            if implausible.maximum is not None and atypical.maximum is not None:
                self.assertLessEqual(
                    atypical.maximum,
                    implausible.maximum,
                    "{}: atypical maximum exceeds implausible maximum".format(
                        feature_range.feature_type
                    ),
                )

    def test_slow_faller_gait_is_atypical_not_implausible(self):
        """Clinically valid slow gait must never be reported as an extraction defect.

        0.295 m/s is the slowest speed in this project's own clinical reference data.
        """
        gait_speed = self.catalog.get(FeatureType.GAIT_SPEED)
        self.assertEqual(ATYPICAL, gait_speed.classify(0.295))
        self.assertEqual(IMPLAUSIBLE, gait_speed.classify(0.0))
        self.assertIsNone(gait_speed.classify(1.15))

    def test_sparc_upper_bound_respects_mathematical_limit(self):
        sparc = self.catalog.get(FeatureType.STRIDE_SPARC)
        self.assertEqual(IMPLAUSIBLE, sparc.classify(-0.5))
        self.assertIsNone(sparc.classify(-2.56))

    def test_normalized_covariance_permits_negative_values(self):
        """The test-suite range wrongly used [0, 500]; real data goes negative."""
        covariance = self.catalog.get(FeatureType.INTRA_STEP_COVARIANCE_V)
        self.assertNotEqual(IMPLAUSIBLE, covariance.classify(-0.675))
        self.assertEqual(IMPLAUSIBLE, covariance.classify(1.5))

    def test_asymmetry_features_use_parent_units_not_ratio(self):
        self.assertEqual("s", self.catalog.get(FeatureType.STRIDE_TIME_ASYMMETRY).unit)
        self.assertEqual("m", self.catalog.get(FeatureType.STEP_LENGTH_ASYMMETRY).unit)
        self.assertEqual("m/s", self.catalog.get(FeatureType.GAIT_SPEED_ASYMMETRY).unit)

    def test_every_defined_tier_has_a_provenance(self):
        for feature_range in self.catalog.ranges.values():
            for tier in (IMPLAUSIBLE, ATYPICAL):
                defined = getattr(feature_range, tier) is not None
                self.assertEqual(
                    defined,
                    bool(feature_range.provenance_for(tier)),
                    "{} {} provenance does not match tier definition".format(
                        feature_range.feature_type, tier
                    ),
                )

    def test_asymmetry_and_epoch_tiers_are_never_labelled_literature(self):
        """Published asymmetry norms use percentages and epoch features have no norms."""
        for feature_range in self.catalog.ranges.values():
            name = feature_range.feature_type.value
            if "asymmetry" in name or (
                name.startswith("epoch:") and "dominant frequency" not in name
            ):
                for tier in (IMPLAUSIBLE, ATYPICAL):
                    self.assertFalse(
                        feature_range.is_literature_based(tier),
                        "{} {} is labelled literature-based".format(name, tier),
                    )

    def test_literature_direct_atypical_tiers_are_the_expected_set(self):
        """Pins the short list of features whose atypical band is anchored on published values."""
        direct = {
            feature_range.feature_type
            for feature_range in self.catalog.ranges.values()
            if feature_range.atypical_provenance == "literature_direct"
        }
        self.assertEqual(
            {
                FeatureType.GAIT_SPEED,
                FeatureType.CADENCE,
                FeatureType.STRIDE_LENGTH,
                FeatureType.HARMONIC_RATIO_V,
                FeatureType.STRIDE_SPARC,
            },
            direct,
        )


class TestCatalogLoaderValidation(unittest.TestCase):
    def test_unknown_feature_type_raises(self):
        path = _write_catalog(
            {"features": [{"feature_type": "not a real feature", "implausible": [0, 1]}]}
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("not a known FeatureType", str(ctx.exception))

    def test_unknown_group_raises(self):
        path = _write_catalog(
            {"features": [{"feature_type": "gait speed", "group": "nonexistent"}]}
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("unknown group", str(ctx.exception))

    def test_unknown_reference_key_raises(self):
        path = _write_catalog(
            {
                "features": [
                    {
                        "feature_type": "gait speed",
                        "implausible": [0, 3],
                        "reference": "ghost_citation",
                    }
                ]
            }
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("unknown reference key", str(ctx.exception))

    def test_inverted_bounds_raise(self):
        path = _write_catalog(
            {"features": [{"feature_type": "gait speed", "implausible": [3.0, 0.5]}]}
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("exceeds maximum", str(ctx.exception))

    def test_duplicate_feature_entry_raises(self):
        path = _write_catalog(
            {
                "features": [
                    {
                        "feature_type": "gait speed",
                        "implausible": [0, 3],
                        "provenance": {"implausible": "judgment"},
                    },
                    {
                        "feature_type": "gait speed",
                        "implausible": [0, 2],
                        "provenance": {"implausible": "judgment"},
                    },
                ]
            }
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("duplicate entry", str(ctx.exception))

    def test_entry_without_any_tier_raises(self):
        path = _write_catalog({"features": [{"feature_type": "gait speed", "unit": "m/s"}]})
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("at least one of", str(ctx.exception))

    def test_group_fields_are_inherited_and_overridable(self):
        path = _write_catalog(
            {
                "groups": {
                    "timing": {
                        "unit": "s",
                        "implausible": [0.1, 2.0],
                        "provenance": {"implausible": "judgment"},
                    }
                },
                "features": [
                    {"feature_type": "stride time", "group": "timing"},
                    {"feature_type": "step time", "group": "timing", "unit": "custom"},
                ],
            }
        )
        catalog = load_range_catalog(path)
        inherited = catalog.get(FeatureType.STRIDE_TIME)
        overridden = catalog.get(FeatureType.STEP_TIME)
        self.assertEqual("s", inherited.unit)
        self.assertEqual(Bounds(0.1, 2.0), inherited.implausible)
        self.assertEqual("custom", overridden.unit)
        self.assertEqual(Bounds(0.1, 2.0), overridden.implausible)
        self.assertEqual("judgment", inherited.implausible_provenance)

    def _single_entry(self, **fields):
        entry = {"feature_type": "gait speed"}
        entry.update(fields)
        return _write_catalog({"features": [entry]})

    def test_tier_without_provenance_raises(self):
        path = self._single_entry(implausible=[0, 3], provenance={})
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("has no provenance", str(ctx.exception))

    def test_missing_provenance_object_raises(self):
        path = self._single_entry(implausible=[0, 3])
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("'provenance' must be an object", str(ctx.exception))

    def test_unknown_provenance_label_raises(self):
        path = self._single_entry(
            implausible=[0, 3], provenance={"implausible": "peer_reviewed_probably"}
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("unknown provenance", str(ctx.exception))

    def test_provenance_for_undefined_tier_raises(self):
        path = self._single_entry(
            implausible=[0, 3],
            provenance={"implausible": "judgment", "atypical": "literature_direct"},
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("undefined tier", str(ctx.exception))

    def test_provenance_naming_unknown_tier_raises(self):
        path = self._single_entry(
            implausible=[0, 3], provenance={"implausible": "judgment", "hard": "judgment"}
        )
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(path)
        self.assertIn("unknown tier", str(ctx.exception))

    def test_missing_file_raises(self):
        with self.assertRaises(RangeCatalogError):
            load_range_catalog(Path("/nonexistent/ranges.json"))

    def test_malformed_json_raises(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        handle.write("{not valid json")
        handle.close()
        with self.assertRaises(RangeCatalogError) as ctx:
            load_range_catalog(Path(handle.name))
        self.assertIn("not valid JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
