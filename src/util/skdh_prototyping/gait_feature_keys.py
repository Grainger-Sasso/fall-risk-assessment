from enum import Enum, auto


class GaitFeatureKeys(Enum):
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
    IC_TIME = "IC Time"

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

    @classmethod
    def get_value(cls, key: str) -> str:
        """Get the string value for a given enum key"""
        return cls[key].value

    @classmethod
    def get_all_values(cls) -> list[str]:
        """Get a list of all string values"""
        return [member.value for member in cls]
