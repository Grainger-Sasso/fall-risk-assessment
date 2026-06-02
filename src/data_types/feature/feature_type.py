from enum import Enum
from typing import Dict, List, Tuple


class FeatureType(Enum):
    """
    Collection of raw feature types
    """

    # Time and Day related features
    DAY_N = "Day N"
    DATE = "Date"
    PROCESSING_START_HOUR = "Processing Start Hour"
    PROCESSING_END_HOUR = "Processing End Hour"
    DAY_START_TIMESTAMP = "Day Start Timestamp"
    DAY_END_TIMESTAMP = "Day End Timestamp"

    # Bout related features
    BOUT_N = "Bout N"
    BOUT_STARTS = "Bout Starts"
    BOUT_DURATION = "Bout Duration"
    BOUT_STEPS = "Bout Steps"
    GAIT_CYCLES = "Gait Cycles"
    # Initial contact time
    IC_TIME = "IC Time"
    # NOTE: THIS IS A DERIVATIVE OF THE IC TIME USING BOUT START AND END INDICES
    BOUT_START_TIMESTAMP = "Bout Start Timestamp"
    BOUT_END_TIMESTAMP = "Bout End Timestamp"

    # Debug features
    DEBUG_MEAN_STEP_FREQ = "debug:mean step freq"
    DEBUG_V_AXIS_EST = "debug:v axis est"
    DEBUG_AP_AXIS_EST = "debug:ap axis est"

    # Basic gait features
    TURN = "Turn"
    STRIDE_TIME = "stride time"
    STRIDE_TIME_ASYMMETRY = "stride time asymmetry"
    STANCE_TIME = "stance time"
    STANCE_TIME_ASYMMETRY = "stance time asymmetry"
    SWING_TIME = "swing time"
    SWING_TIME_ASYMMETRY = "swing time asymmetry"
    STEP_TIME = "step time"
    STEP_TIME_ASYMMETRY = "step time asymmetry"

    # Support phase features
    INITIAL_DOUBLE_SUPPORT = "initial double support"
    INITIAL_DOUBLE_SUPPORT_ASYMMETRY = "initial double support asymmetry"
    TERMINAL_DOUBLE_SUPPORT = "terminal double support"
    TERMINAL_DOUBLE_SUPPORT_ASYMMETRY = "terminal double support asymmetry"
    DOUBLE_SUPPORT = "double support"
    DOUBLE_SUPPORT_ASYMMETRY = "double support asymmetry"
    SINGLE_SUPPORT = "single support"
    SINGLE_SUPPORT_ASYMMETRY = "single support asymmetry"

    # Distance and speed features
    M2_DELTA_H = "m2 delta h"
    M2_DELTA_H_PRIME = "m2 delta h prime"
    STEP_LENGTH = "step length"
    STEP_LENGTH_ASYMMETRY = "step length asymmetry"
    STRIDE_LENGTH = "stride length"
    STRIDE_LENGTH_ASYMMETRY = "stride length asymmetry"
    GAIT_SPEED = "gait speed"
    GAIT_SPEED_ASYMMETRY = "gait speed asymmetry"
    CADENCE = "cadence"

    # M1-specific features
    M1_DELTA_H = "m1 delta h"
    STEP_LENGTH_M1 = "step length m1"
    STEP_LENGTH_M1_ASYMMETRY = "step length m1 asymmetry"
    STRIDE_LENGTH_M1 = "stride length m1"
    STRIDE_LENGTH_M1_ASYMMETRY = "stride length m1 asymmetry"
    GAIT_SPEED_M1 = "gait speed m1"
    GAIT_SPEED_M1_ASYMMETRY = "gait speed m1 asymmetry"

    # Advanced gait metrics
    INTRA_STEP_COVARIANCE_V = "intra-step covariance - V"
    INTRA_STRIDE_COVARIANCE_V = "intra-stride covariance - V"
    HARMONIC_RATIO_V = "harmonic ratio - V"
    STRIDE_SPARC = "stride SPARC"

    # Bout-level metrics
    BOUT_PHASE_COORDINATION_INDEX = "bout:phase coordination index"
    BOUT_GAIT_SYMMETRY_INDEX = "bout:gait symmetry index"
    BOUT_STEP_REGULARITY_V = "bout:step regularity - V"
    BOUT_STRIDE_REGULARITY_V = "bout:stride regularity - V"
    BOUT_AUTOCOVARIANCE_SYMMETRY_V = "bout:autocovariance symmetry - V"
    BOUT_REGULARITY_INDEX_V = "bout:regularity index - V"

    # Epoch basis features (sliding windows)
    EPOCH_VERTICAL_MEAN = "epoch:vertical mean"
    EPOCH_VERTICAL_STD = "epoch:vertical std"
    EPOCH_VERTICAL_RMS = "epoch:vertical rms"
    EPOCH_VERTICAL_SMA = "epoch:vertical sma"
    EPOCH_VERTICAL_JERK_RMS = "epoch:vertical jerk rms"
    EPOCH_VERTICAL_ZCR = "epoch:vertical zcr"
    EPOCH_VERTICAL_DOMINANT_FREQUENCY = "epoch:vertical dominant frequency"
    EPOCH_VERTICAL_SPECTRAL_ENTROPY = "epoch:vertical spectral entropy"
    EPOCH_VERTICAL_BANDPOWER_LOW = "epoch:vertical bandpower low"
    EPOCH_VERTICAL_BANDPOWER_HIGH = "epoch:vertical bandpower high"

    EPOCH_MEDIOLATERAL_MEAN = "epoch:mediolateral mean"
    EPOCH_MEDIOLATERAL_STD = "epoch:mediolateral std"
    EPOCH_MEDIOLATERAL_RMS = "epoch:mediolateral rms"
    EPOCH_MEDIOLATERAL_SMA = "epoch:mediolateral sma"
    EPOCH_MEDIOLATERAL_JERK_RMS = "epoch:mediolateral jerk rms"
    EPOCH_MEDIOLATERAL_ZCR = "epoch:mediolateral zcr"
    EPOCH_MEDIOLATERAL_DOMINANT_FREQUENCY = "epoch:mediolateral dominant frequency"
    EPOCH_MEDIOLATERAL_SPECTRAL_ENTROPY = "epoch:mediolateral spectral entropy"
    EPOCH_MEDIOLATERAL_BANDPOWER_LOW = "epoch:mediolateral bandpower low"
    EPOCH_MEDIOLATERAL_BANDPOWER_HIGH = "epoch:mediolateral bandpower high"

    EPOCH_ANTEROPOSTERIOR_MEAN = "epoch:anteroposterior mean"
    EPOCH_ANTEROPOSTERIOR_STD = "epoch:anteroposterior std"
    EPOCH_ANTEROPOSTERIOR_RMS = "epoch:anteroposterior rms"
    EPOCH_ANTEROPOSTERIOR_SMA = "epoch:anteroposterior sma"
    EPOCH_ANTEROPOSTERIOR_JERK_RMS = "epoch:anteroposterior jerk rms"
    EPOCH_ANTEROPOSTERIOR_ZCR = "epoch:anteroposterior zcr"
    EPOCH_ANTEROPOSTERIOR_DOMINANT_FREQUENCY = "epoch:anteroposterior dominant frequency"
    EPOCH_ANTEROPOSTERIOR_SPECTRAL_ENTROPY = "epoch:anteroposterior spectral entropy"
    EPOCH_ANTEROPOSTERIOR_BANDPOWER_LOW = "epoch:anteroposterior bandpower low"
    EPOCH_ANTEROPOSTERIOR_BANDPOWER_HIGH = "epoch:anteroposterior bandpower high"

    EPOCH_MAGNITUDE_MEAN = "epoch:magnitude mean"
    EPOCH_MAGNITUDE_STD = "epoch:magnitude std"
    EPOCH_MAGNITUDE_RMS = "epoch:magnitude rms"
    EPOCH_MAGNITUDE_SMA = "epoch:magnitude sma"
    EPOCH_MAGNITUDE_JERK_RMS = "epoch:magnitude jerk rms"
    EPOCH_MAGNITUDE_ZCR = "epoch:magnitude zcr"
    EPOCH_MAGNITUDE_DOMINANT_FREQUENCY = "epoch:magnitude dominant frequency"
    EPOCH_MAGNITUDE_SPECTRAL_ENTROPY = "epoch:magnitude spectral entropy"
    EPOCH_MAGNITUDE_BANDPOWER_LOW = "epoch:magnitude bandpower low"
    EPOCH_MAGNITUDE_BANDPOWER_HIGH = "epoch:magnitude bandpower high"

    @classmethod
    def get_value(cls, key: str) -> str:
        """Get the string value for a given enum key"""
        return cls[key].value

    @classmethod
    def get_all_values(cls) -> list[str]:
        """Get a list of all string values"""
        return [member.value for member in cls]

    @classmethod
    def get_stride_feature_types(cls) -> List["FeatureType"]:
        return [
            cls.STRIDE_TIME,
            cls.STRIDE_TIME_ASYMMETRY,
            cls.STANCE_TIME,
            cls.STANCE_TIME_ASYMMETRY,
            cls.SWING_TIME,
            cls.SWING_TIME_ASYMMETRY,
            cls.STEP_TIME,
            cls.STEP_TIME_ASYMMETRY,
            cls.INITIAL_DOUBLE_SUPPORT,
            cls.INITIAL_DOUBLE_SUPPORT_ASYMMETRY,
            cls.TERMINAL_DOUBLE_SUPPORT,
            cls.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY,
            cls.DOUBLE_SUPPORT,
            cls.DOUBLE_SUPPORT_ASYMMETRY,
            cls.SINGLE_SUPPORT,
            cls.SINGLE_SUPPORT_ASYMMETRY,
            cls.M2_DELTA_H,
            cls.M2_DELTA_H_PRIME,
            cls.STEP_LENGTH,
            cls.STEP_LENGTH_ASYMMETRY,
            cls.STRIDE_LENGTH,
            cls.STRIDE_LENGTH_ASYMMETRY,
            cls.GAIT_SPEED,
            cls.GAIT_SPEED_ASYMMETRY,
            cls.CADENCE,
            cls.M1_DELTA_H,
            cls.STEP_LENGTH_M1,
            cls.STEP_LENGTH_M1_ASYMMETRY,
            cls.STRIDE_LENGTH_M1,
            cls.STRIDE_LENGTH_M1_ASYMMETRY,
            cls.GAIT_SPEED_M1,
            cls.GAIT_SPEED_M1_ASYMMETRY,
            cls.INTRA_STEP_COVARIANCE_V,
            cls.INTRA_STRIDE_COVARIANCE_V,
            cls.HARMONIC_RATIO_V,
            cls.STRIDE_SPARC,
        ]

    @classmethod
    def get_epoch_feature_types(cls) -> List["FeatureType"]:
        mapping = cls._epoch_feature_map()
        ordered_keys: List[Tuple[str, str]] = []
        for axis_key in ("vertical", "mediolateral", "anteroposterior", "magnitude"):
            for metric_name in (
                "mean",
                "std",
                "rms",
                "sma",
                "jerk_rms",
                "zero_crossing_rate",
                "dominant_frequency",
                "spectral_entropy",
                "bandpower_low",
                "bandpower_high",
            ):
                ordered_keys.append((axis_key, metric_name))
        return [mapping[key] for key in ordered_keys]

    @classmethod
    def get_epoch_feature_type(cls, axis_key: str, metric_name: str) -> "FeatureType":
        mapping = cls._epoch_feature_map()
        key = (axis_key, metric_name)
        if key not in mapping:
            raise ValueError(f"Unsupported epoch feature selector: {axis_key}/{metric_name}")
        return mapping[key]

    @classmethod
    def _epoch_feature_map(cls) -> Dict[Tuple[str, str], "FeatureType"]:
        return {
            ("vertical", "mean"): cls.EPOCH_VERTICAL_MEAN,
            ("vertical", "std"): cls.EPOCH_VERTICAL_STD,
            ("vertical", "rms"): cls.EPOCH_VERTICAL_RMS,
            ("vertical", "sma"): cls.EPOCH_VERTICAL_SMA,
            ("vertical", "jerk_rms"): cls.EPOCH_VERTICAL_JERK_RMS,
            ("vertical", "zero_crossing_rate"): cls.EPOCH_VERTICAL_ZCR,
            ("vertical", "dominant_frequency"): cls.EPOCH_VERTICAL_DOMINANT_FREQUENCY,
            ("vertical", "spectral_entropy"): cls.EPOCH_VERTICAL_SPECTRAL_ENTROPY,
            ("vertical", "bandpower_low"): cls.EPOCH_VERTICAL_BANDPOWER_LOW,
            ("vertical", "bandpower_high"): cls.EPOCH_VERTICAL_BANDPOWER_HIGH,
            ("mediolateral", "mean"): cls.EPOCH_MEDIOLATERAL_MEAN,
            ("mediolateral", "std"): cls.EPOCH_MEDIOLATERAL_STD,
            ("mediolateral", "rms"): cls.EPOCH_MEDIOLATERAL_RMS,
            ("mediolateral", "sma"): cls.EPOCH_MEDIOLATERAL_SMA,
            ("mediolateral", "jerk_rms"): cls.EPOCH_MEDIOLATERAL_JERK_RMS,
            ("mediolateral", "zero_crossing_rate"): cls.EPOCH_MEDIOLATERAL_ZCR,
            (
                "mediolateral",
                "dominant_frequency",
            ): cls.EPOCH_MEDIOLATERAL_DOMINANT_FREQUENCY,
            ("mediolateral", "spectral_entropy"): cls.EPOCH_MEDIOLATERAL_SPECTRAL_ENTROPY,
            ("mediolateral", "bandpower_low"): cls.EPOCH_MEDIOLATERAL_BANDPOWER_LOW,
            ("mediolateral", "bandpower_high"): cls.EPOCH_MEDIOLATERAL_BANDPOWER_HIGH,
            ("anteroposterior", "mean"): cls.EPOCH_ANTEROPOSTERIOR_MEAN,
            ("anteroposterior", "std"): cls.EPOCH_ANTEROPOSTERIOR_STD,
            ("anteroposterior", "rms"): cls.EPOCH_ANTEROPOSTERIOR_RMS,
            ("anteroposterior", "sma"): cls.EPOCH_ANTEROPOSTERIOR_SMA,
            ("anteroposterior", "jerk_rms"): cls.EPOCH_ANTEROPOSTERIOR_JERK_RMS,
            ("anteroposterior", "zero_crossing_rate"): cls.EPOCH_ANTEROPOSTERIOR_ZCR,
            (
                "anteroposterior",
                "dominant_frequency",
            ): cls.EPOCH_ANTEROPOSTERIOR_DOMINANT_FREQUENCY,
            ("anteroposterior", "spectral_entropy"): cls.EPOCH_ANTEROPOSTERIOR_SPECTRAL_ENTROPY,
            ("anteroposterior", "bandpower_low"): cls.EPOCH_ANTEROPOSTERIOR_BANDPOWER_LOW,
            ("anteroposterior", "bandpower_high"): cls.EPOCH_ANTEROPOSTERIOR_BANDPOWER_HIGH,
            ("magnitude", "mean"): cls.EPOCH_MAGNITUDE_MEAN,
            ("magnitude", "std"): cls.EPOCH_MAGNITUDE_STD,
            ("magnitude", "rms"): cls.EPOCH_MAGNITUDE_RMS,
            ("magnitude", "sma"): cls.EPOCH_MAGNITUDE_SMA,
            ("magnitude", "jerk_rms"): cls.EPOCH_MAGNITUDE_JERK_RMS,
            ("magnitude", "zero_crossing_rate"): cls.EPOCH_MAGNITUDE_ZCR,
            ("magnitude", "dominant_frequency"): cls.EPOCH_MAGNITUDE_DOMINANT_FREQUENCY,
            ("magnitude", "spectral_entropy"): cls.EPOCH_MAGNITUDE_SPECTRAL_ENTROPY,
            ("magnitude", "bandpower_low"): cls.EPOCH_MAGNITUDE_BANDPOWER_LOW,
            ("magnitude", "bandpower_high"): cls.EPOCH_MAGNITUDE_BANDPOWER_HIGH,
        }
