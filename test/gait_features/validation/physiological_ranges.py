"""Physiological plausibility bounds for gait stride and epoch features.

These ranges apply only to finite (non-NaN) values on usable sample rows.
They are intentionally permissive for older-adult free-living and treadmill
walking, but reject obvious extraction failures (zeros, sign errors, units bugs).

Ranges cite typical adult walking literature (cadence ~100-120 steps/min,
stride time ~1.0 s, gait speed ~1.0-1.4 m/s) with margin for slow gait and
lab/treadmill conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.data_types.feature.feature_type import FeatureType


@dataclass(frozen=True)
class PhysiologicalRange:
    """Inclusive bounds for a single feature type."""

    description: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    strictly_positive: bool = False

    def contains(self, value: float) -> bool:
        if not (value == value):  # NaN
            return True
        if self.strictly_positive and value <= 0.0:
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


# Shared bounds reused across feature groups.
_TIME_SYMMETRY = PhysiologicalRange(
    "Temporal gait phase duration during detected walking.",
    minimum=0.15,
    maximum=2.5,
    strictly_positive=True,
)
_ASYMMETRY = PhysiologicalRange(
    "Left-right asymmetry ratio; near zero is symmetric, |value| > 1 is unusual.",
    minimum=-1.5,
    maximum=1.5,
)
_LENGTH = PhysiologicalRange(
    "Step/stride length during walking.",
    minimum=0.05,
    maximum=2.0,
    strictly_positive=True,
)
_SPEED = PhysiologicalRange(
    "Walking speed; 0 m/s on a detected walking stride indicates extraction failure.",
    minimum=0.15,
    maximum=3.0,
    strictly_positive=True,
)

STRIDE_PHYSIOLOGICAL_RANGES: Dict[FeatureType, PhysiologicalRange] = {
    FeatureType.STRIDE_TIME: PhysiologicalRange(
        "Heel-strike to heel-strike duration; adult walking ~0.9-1.2 s.",
        minimum=0.4,
        maximum=2.5,
        strictly_positive=True,
    ),
    FeatureType.STRIDE_TIME_ASYMMETRY: _ASYMMETRY,
    FeatureType.STANCE_TIME: _TIME_SYMMETRY,
    FeatureType.STANCE_TIME_ASYMMETRY: _ASYMMETRY,
    FeatureType.SWING_TIME: PhysiologicalRange(
        "Swing phase duration; typically ~35-40% of stride time.",
        minimum=0.15,
        maximum=1.5,
        strictly_positive=True,
    ),
    FeatureType.SWING_TIME_ASYMMETRY: _ASYMMETRY,
    FeatureType.STEP_TIME: PhysiologicalRange(
        "Consecutive foot-contact interval; roughly half of stride time.",
        minimum=0.2,
        maximum=1.5,
        strictly_positive=True,
    ),
    FeatureType.STEP_TIME_ASYMMETRY: _ASYMMETRY,
    FeatureType.INITIAL_DOUBLE_SUPPORT: PhysiologicalRange(
        "Initial double-support duration.",
        minimum=0.02,
        maximum=0.6,
        strictly_positive=True,
    ),
    FeatureType.INITIAL_DOUBLE_SUPPORT_ASYMMETRY: _ASYMMETRY,
    FeatureType.TERMINAL_DOUBLE_SUPPORT: PhysiologicalRange(
        "Terminal double-support duration.",
        minimum=0.02,
        maximum=0.6,
        strictly_positive=True,
    ),
    FeatureType.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY: _ASYMMETRY,
    FeatureType.DOUBLE_SUPPORT: PhysiologicalRange(
        "Total double-support duration per stride.",
        minimum=0.05,
        maximum=0.8,
        strictly_positive=True,
    ),
    FeatureType.DOUBLE_SUPPORT_ASYMMETRY: _ASYMMETRY,
    FeatureType.SINGLE_SUPPORT: PhysiologicalRange(
        "Single-limb support duration.",
        minimum=0.15,
        maximum=1.5,
        strictly_positive=True,
    ),
    FeatureType.SINGLE_SUPPORT_ASYMMETRY: _ASYMMETRY,
    FeatureType.M2_DELTA_H: PhysiologicalRange(
        "Vertical pelvis displacement (method 2); walking typically 1-5 cm.",
        minimum=0.001,
        maximum=0.25,
        strictly_positive=True,
    ),
    FeatureType.M2_DELTA_H_PRIME: PhysiologicalRange(
        "Normalized vertical pelvis displacement (method 2).",
        minimum=0.001,
        maximum=0.25,
        strictly_positive=True,
    ),
    FeatureType.STEP_LENGTH: _LENGTH,
    FeatureType.STEP_LENGTH_ASYMMETRY: _ASYMMETRY,
    FeatureType.STRIDE_LENGTH: PhysiologicalRange(
        "Stride length; typically ~1.0-1.4 m in healthy walking.",
        minimum=0.1,
        maximum=2.5,
        strictly_positive=True,
    ),
    FeatureType.STRIDE_LENGTH_ASYMMETRY: _ASYMMETRY,
    FeatureType.GAIT_SPEED: _SPEED,
    FeatureType.GAIT_SPEED_ASYMMETRY: _ASYMMETRY,
    FeatureType.CADENCE: PhysiologicalRange(
        "Steps per minute; adult walking ~90-130, slow gait may be lower.",
        minimum=20.0,
        maximum=200.0,
        strictly_positive=True,
    ),
    FeatureType.M1_DELTA_H: PhysiologicalRange(
        "Vertical pelvis displacement (method 1).",
        minimum=0.001,
        maximum=0.25,
        strictly_positive=True,
    ),
    FeatureType.STEP_LENGTH_M1: _LENGTH,
    FeatureType.STEP_LENGTH_M1_ASYMMETRY: _ASYMMETRY,
    FeatureType.STRIDE_LENGTH_M1: PhysiologicalRange(
        "Stride length (method 1).",
        minimum=0.1,
        maximum=2.5,
        strictly_positive=True,
    ),
    FeatureType.STRIDE_LENGTH_M1_ASYMMETRY: _ASYMMETRY,
    FeatureType.GAIT_SPEED_M1: _SPEED,
    FeatureType.GAIT_SPEED_M1_ASYMMETRY: _ASYMMETRY,
    FeatureType.INTRA_STEP_COVARIANCE_V: PhysiologicalRange(
        "Vertical intra-step acceleration covariance; should be non-negative.",
        minimum=0.0,
        maximum=500.0,
    ),
    FeatureType.INTRA_STRIDE_COVARIANCE_V: PhysiologicalRange(
        "Vertical intra-stride acceleration covariance; should be non-negative.",
        minimum=0.0,
        maximum=500.0,
    ),
    FeatureType.HARMONIC_RATIO_V: PhysiologicalRange(
        "Vertical harmonic ratio; walking typically > 1, harmonics can push higher.",
        minimum=0.5,
        maximum=12.0,
        strictly_positive=True,
    ),
    FeatureType.STRIDE_SPARC: PhysiologicalRange(
        "Spectral arc length of vertical acceleration; smoother gait is less negative.",
        minimum=-20.0,
        maximum=0.0,
    ),
}

_EPOCH_ACCEL_MEAN = PhysiologicalRange(
    "Per-axis acceleration mean in sensor frame (m/s²); AP/ML near 0, "
    "sensor Z may not align with gravity so vertical mean can be signed.",
    minimum=-15.0,
    maximum=15.0,
)
_EPOCH_ACCEL_DISPERSION = PhysiologicalRange(
    "Acceleration variability (std/rms/sma) during walking; non-zero movement.",
    minimum=0.01,
    maximum=15.0,
    strictly_positive=True,
)
_EPOCH_JERK_RMS = PhysiologicalRange(
    "Jerk RMS (m/s³) from finite-difference at sampling rate; lumbar walking "
    "typically tens to low hundreds depending on sample rate.",
    minimum=0.1,
    maximum=500.0,
    strictly_positive=True,
)
_EPOCH_ZCR = PhysiologicalRange(
    "Zero-crossing rate of detrended acceleration (1/s).",
    minimum=0.0,
    maximum=10.0,
)
_EPOCH_DOM_FREQ = PhysiologicalRange(
    "Dominant frequency in gait band; walking fundamental ~0.8-2.5 Hz, "
    "harmonics and treadmill cadence can reach higher.",
    minimum=0.3,
    maximum=10.0,
    strictly_positive=True,
)
_EPOCH_SPECTRAL_ENTROPY = PhysiologicalRange(
    "Spectral entropy (bits); non-negative.",
    minimum=0.0,
    maximum=20.0,
)
_EPOCH_BANDPOWER = PhysiologicalRange(
    "Integrated power in gait frequency band; non-negative.",
    minimum=0.0,
    maximum=1e6,
)

EPOCH_PHYSIOLOGICAL_RANGES: Dict[FeatureType, PhysiologicalRange] = {}

for _axis in ("vertical", "mediolateral", "anteroposterior"):
    EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.get_epoch_feature_type(_axis, "mean")] = (
        _EPOCH_ACCEL_MEAN
    )
    for _metric in ("std", "rms", "sma"):
        EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.get_epoch_feature_type(_axis, _metric)] = (
            _EPOCH_ACCEL_DISPERSION
        )
    EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.get_epoch_feature_type(_axis, "jerk_rms")] = (
        _EPOCH_JERK_RMS
    )
    EPOCH_PHYSIOLOGICAL_RANGES[
        FeatureType.get_epoch_feature_type(_axis, "zero_crossing_rate")
    ] = _EPOCH_ZCR
    EPOCH_PHYSIOLOGICAL_RANGES[
        FeatureType.get_epoch_feature_type(_axis, "dominant_frequency")
    ] = _EPOCH_DOM_FREQ
    EPOCH_PHYSIOLOGICAL_RANGES[
        FeatureType.get_epoch_feature_type(_axis, "spectral_entropy")
    ] = _EPOCH_SPECTRAL_ENTROPY
    for _band in ("bandpower_low", "bandpower_high"):
        EPOCH_PHYSIOLOGICAL_RANGES[
            FeatureType.get_epoch_feature_type(_axis, _band)
        ] = _EPOCH_BANDPOWER

# Magnitude axis: vector norm, so mean should stay near 1g during walking.
EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.EPOCH_MAGNITUDE_MEAN] = PhysiologicalRange(
    "Acceleration magnitude mean; lumbar walking typically ~9-11 m/s².",
    minimum=5.0,
    maximum=15.0,
    strictly_positive=True,
)
for _metric in ("std", "rms", "sma"):
    EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.get_epoch_feature_type("magnitude", _metric)] = (
        _EPOCH_ACCEL_DISPERSION
    )
EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.get_epoch_feature_type("magnitude", "jerk_rms")] = (
    _EPOCH_JERK_RMS
)
EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.EPOCH_MAGNITUDE_ZCR] = _EPOCH_ZCR
EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.EPOCH_MAGNITUDE_DOMINANT_FREQUENCY] = _EPOCH_DOM_FREQ
EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.EPOCH_MAGNITUDE_SPECTRAL_ENTROPY] = _EPOCH_SPECTRAL_ENTROPY
for _band in ("bandpower_low", "bandpower_high"):
    EPOCH_PHYSIOLOGICAL_RANGES[FeatureType.get_epoch_feature_type("magnitude", _band)] = (
        _EPOCH_BANDPOWER
    )

# Key features that must be populated (finite and non-zero) for a stride to be meaningful.
STRIDE_POPULATION_ANCHORS = (
    FeatureType.STRIDE_TIME,
    FeatureType.GAIT_SPEED,
    FeatureType.CADENCE,
)

# Key epoch features that confirm accelerometer variability in a walking window.
EPOCH_POPULATION_ANCHORS = (
    FeatureType.EPOCH_MAGNITUDE_RMS,
    FeatureType.EPOCH_VERTICAL_RMS,
    FeatureType.EPOCH_MAGNITUDE_MEAN,
)
