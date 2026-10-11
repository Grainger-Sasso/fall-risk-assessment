"""Load and resolve the physiological range catalog used by post-hoc validation.

The catalog is JSON so that ranges and their citations can be reviewed and edited
without touching code. This module resolves group inheritance and reference keys,
and validates every entry against the ``FeatureType`` enum so a typo in the JSON
fails loudly at load time rather than silently skipping a feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data_types.feature.feature_type import FeatureType

DEFAULT_CATALOG_PATH = Path(__file__).parent / "ranges" / "gait_feature_ranges.json"

IMPLAUSIBLE = "implausible"
ATYPICAL = "atypical"
TIERS = (IMPLAUSIBLE, ATYPICAL)

LITERATURE_DIRECT = "literature_direct"
LITERATURE_DERIVED = "literature_derived"
PHYSICAL_LIMIT = "physical_limit"
EMPIRICAL = "empirical"
JUDGMENT = "judgment"
PROVENANCE_VALUES = (
    LITERATURE_DIRECT,
    LITERATURE_DERIVED,
    PHYSICAL_LIMIT,
    EMPIRICAL,
    JUDGMENT,
)
LITERATURE_PROVENANCES = frozenset({LITERATURE_DIRECT, LITERATURE_DERIVED})


class RangeCatalogError(ValueError):
    """Raised when the range catalog is malformed or references unknown names."""


@dataclass(frozen=True)
class Bounds:
    """Inclusive bounds where ``None`` means unbounded on that side."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def contains(self, value: float) -> bool:
        if value != value:  # NaN is a missingness concern, not a range violation
            return True
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def describe(self) -> str:
        low = "-inf" if self.minimum is None else "{:g}".format(self.minimum)
        high = "+inf" if self.maximum is None else "{:g}".format(self.maximum)
        return "[{}, {}]".format(low, high)


@dataclass(frozen=True)
class FeatureRange:
    """Two-tier range definition for a single feature type."""

    feature_type: FeatureType
    unit: str
    reference_key: str
    citation: str
    note: str
    implausible: Optional[Bounds] = None
    atypical: Optional[Bounds] = None
    implausible_provenance: str = ""
    atypical_provenance: str = ""

    def provenance_for(self, tier: str) -> str:
        """Provenance label of ``tier``, or an empty string when the tier is undefined."""
        if tier == IMPLAUSIBLE:
            return self.implausible_provenance
        if tier == ATYPICAL:
            return self.atypical_provenance
        raise ValueError("Unknown tier: {}".format(tier))

    def is_literature_based(self, tier: str) -> bool:
        return self.provenance_for(tier) in LITERATURE_PROVENANCES

    def classify(self, value: float) -> Optional[str]:
        """Return ``IMPLAUSIBLE``, ``ATYPICAL`` or ``None`` for an in-range value.

        NaN returns ``None``: missingness is counted separately so that a padded or
        undefined value is never double-reported as a range violation.
        """
        if value != value:
            return None
        if self.implausible is not None and not self.implausible.contains(value):
            return IMPLAUSIBLE
        if self.atypical is not None and not self.atypical.contains(value):
            return ATYPICAL
        return None


@dataclass(frozen=True)
class RangeCatalog:
    """Resolved range definitions plus the reference metadata behind them."""

    schema_version: str
    ranges: Dict[FeatureType, FeatureRange]
    references: Dict[str, Dict[str, str]]

    def get(self, feature_type: FeatureType) -> Optional[FeatureRange]:
        return self.ranges.get(feature_type)

    def covered_feature_types(self) -> List[FeatureType]:
        return list(self.ranges)

    def missing_coverage(self, feature_types: List[FeatureType]) -> List[FeatureType]:
        """Feature types present in data but absent from the catalog."""
        return [ftype for ftype in feature_types if ftype not in self.ranges]

    def provenance_counts(self, tier: str) -> Dict[str, int]:
        """Number of feature types per provenance label for ``tier``."""
        counts: Dict[str, int] = {}
        for feature_range in self.ranges.values():
            label = feature_range.provenance_for(tier)
            if label:
                counts[label] = counts.get(label, 0) + 1
        return counts


def _parse_bounds(raw: object, context: str) -> Optional[Bounds]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RangeCatalogError(
            "{}: bounds must be a two-element [minimum, maximum] list, got {!r}".format(
                context, raw
            )
        )
    minimum, maximum = raw
    for bound in (minimum, maximum):
        if bound is not None and not isinstance(bound, (int, float)):
            raise RangeCatalogError(
                "{}: bound values must be numbers or null, got {!r}".format(context, bound)
            )
    if minimum is not None and maximum is not None and minimum > maximum:
        raise RangeCatalogError(
            "{}: minimum {} exceeds maximum {}".format(context, minimum, maximum)
        )
    return Bounds(
        minimum=None if minimum is None else float(minimum),
        maximum=None if maximum is None else float(maximum),
    )


def _resolve_feature_type(value: str, context: str) -> FeatureType:
    try:
        return FeatureType(value)
    except ValueError as exc:
        raise RangeCatalogError(
            "{}: '{}' is not a known FeatureType value".format(context, value)
        ) from exc


def _merge_entry(
    entry: Dict[str, object], groups: Dict[str, Dict[str, object]], context: str
) -> Dict[str, object]:
    group_name = entry.get("group")
    if group_name is None:
        return dict(entry)
    if not isinstance(group_name, str) or group_name not in groups:
        raise RangeCatalogError(
            "{}: unknown group '{}'".format(context, group_name)
        )
    merged: Dict[str, object] = dict(groups[group_name])
    merged.update({k: v for k, v in entry.items() if k != "group"})
    return merged


def _parse_provenance(
    raw: object, defined_tiers: Dict[str, Optional[Bounds]], context: str
) -> Dict[str, str]:
    """Require a known provenance label for every defined tier, and only those."""
    if not isinstance(raw, dict):
        raise RangeCatalogError(
            "{}: 'provenance' must be an object mapping tier to label".format(context)
        )
    unknown_tiers = set(raw) - set(TIERS)
    if unknown_tiers:
        raise RangeCatalogError(
            "{}: provenance names unknown tier(s) {}".format(context, sorted(unknown_tiers))
        )

    labels: Dict[str, str] = {}
    for tier, bounds in defined_tiers.items():
        label = raw.get(tier)
        if bounds is None:
            if label is not None:
                raise RangeCatalogError(
                    "{}: provenance given for undefined tier '{}'".format(context, tier)
                )
            continue
        if label is None:
            raise RangeCatalogError(
                "{}: tier '{}' has no provenance".format(context, tier)
            )
        if label not in PROVENANCE_VALUES:
            raise RangeCatalogError(
                "{}: unknown provenance '{}' for tier '{}'; expected one of {}".format(
                    context, label, tier, list(PROVENANCE_VALUES)
                )
            )
        labels[tier] = str(label)
    return labels


def load_range_catalog(path: Optional[Path] = None) -> RangeCatalog:
    """Load, resolve and validate the range catalog at ``path``."""
    catalog_path = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    if not catalog_path.is_file():
        raise RangeCatalogError("Range catalog not found: {}".format(catalog_path))

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RangeCatalogError(
            "Range catalog {} is not valid JSON: {}".format(catalog_path, exc)
        ) from exc

    references = payload.get("references", {})
    groups = payload.get("groups", {})
    entries = payload.get("features")
    if not isinstance(entries, list) or not entries:
        raise RangeCatalogError(
            "Range catalog {} must contain a non-empty 'features' list".format(catalog_path)
        )

    ranges: Dict[FeatureType, FeatureRange] = {}
    for index, raw_entry in enumerate(entries):
        context = "features[{}]".format(index)
        if not isinstance(raw_entry, dict):
            raise RangeCatalogError("{}: entry must be an object".format(context))
        raw_name = raw_entry.get("feature_type")
        if not isinstance(raw_name, str):
            raise RangeCatalogError("{}: missing 'feature_type'".format(context))
        context = "features[{}] ('{}')".format(index, raw_name)

        entry = _merge_entry(raw_entry, groups, context)
        feature_type = _resolve_feature_type(raw_name, context)
        if feature_type in ranges:
            raise RangeCatalogError("{}: duplicate entry for this feature type".format(context))

        reference_key = str(entry.get("reference", ""))
        if reference_key and reference_key not in references:
            raise RangeCatalogError(
                "{}: unknown reference key '{}'".format(context, reference_key)
            )
        reference = references.get(reference_key, {})

        implausible = _parse_bounds(entry.get(IMPLAUSIBLE), "{} implausible".format(context))
        atypical = _parse_bounds(entry.get(ATYPICAL), "{} atypical".format(context))
        if implausible is None and atypical is None:
            raise RangeCatalogError(
                "{}: at least one of 'implausible' or 'atypical' must be defined".format(context)
            )
        provenance = _parse_provenance(
            entry.get("provenance"),
            {IMPLAUSIBLE: implausible, ATYPICAL: atypical},
            context,
        )

        ranges[feature_type] = FeatureRange(
            feature_type=feature_type,
            unit=str(entry.get("unit", "")),
            reference_key=reference_key,
            citation=str(reference.get("citation", "")),
            note=str(entry.get("note", "")),
            implausible=implausible,
            atypical=atypical,
            implausible_provenance=provenance.get(IMPLAUSIBLE, ""),
            atypical_provenance=provenance.get(ATYPICAL, ""),
        )

    return RangeCatalog(
        schema_version=str(payload.get("schema_version", "")),
        ranges=ranges,
        references=references,
    )


@lru_cache(maxsize=4)
def _cached_catalog(path_str: str) -> RangeCatalog:
    return load_range_catalog(Path(path_str))


def default_range_catalog() -> RangeCatalog:
    """Return the bundled catalog, cached across calls."""
    return _cached_catalog(str(DEFAULT_CATALOG_PATH))


def coverage_summary(catalog: RangeCatalog) -> Tuple[int, int]:
    """Return ``(features_with_implausible, features_with_atypical)`` counts."""
    with_implausible = sum(1 for r in catalog.ranges.values() if r.implausible is not None)
    with_atypical = sum(1 for r in catalog.ranges.values() if r.atypical is not None)
    return with_implausible, with_atypical
