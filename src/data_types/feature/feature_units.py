from typing import Dict

from src.data_types.feature.feature_type import FeatureType


STRIDE_DEFAULT_UNIT_MAP: Dict[FeatureType, str] = {
    FeatureType.STRIDE_TIME: "s",
    FeatureType.STRIDE_TIME_ASYMMETRY: "ratio",
    FeatureType.STANCE_TIME: "s",
    FeatureType.STANCE_TIME_ASYMMETRY: "ratio",
    FeatureType.SWING_TIME: "s",
    FeatureType.SWING_TIME_ASYMMETRY: "ratio",
    FeatureType.STEP_TIME: "s",
    FeatureType.STEP_TIME_ASYMMETRY: "ratio",
    FeatureType.INITIAL_DOUBLE_SUPPORT: "s",
    FeatureType.INITIAL_DOUBLE_SUPPORT_ASYMMETRY: "ratio",
    FeatureType.TERMINAL_DOUBLE_SUPPORT: "s",
    FeatureType.TERMINAL_DOUBLE_SUPPORT_ASYMMETRY: "ratio",
    FeatureType.DOUBLE_SUPPORT: "s",
    FeatureType.DOUBLE_SUPPORT_ASYMMETRY: "ratio",
    FeatureType.SINGLE_SUPPORT: "s",
    FeatureType.SINGLE_SUPPORT_ASYMMETRY: "ratio",
    FeatureType.M2_DELTA_H: "m",
    FeatureType.M2_DELTA_H_PRIME: "m",
    FeatureType.STEP_LENGTH: "m",
    FeatureType.STEP_LENGTH_ASYMMETRY: "ratio",
    FeatureType.STRIDE_LENGTH: "m",
    FeatureType.STRIDE_LENGTH_ASYMMETRY: "ratio",
    FeatureType.GAIT_SPEED: "m/s",
    FeatureType.GAIT_SPEED_ASYMMETRY: "ratio",
    FeatureType.CADENCE: "steps/min",
    FeatureType.M1_DELTA_H: "m",
    FeatureType.STEP_LENGTH_M1: "m",
    FeatureType.STEP_LENGTH_M1_ASYMMETRY: "ratio",
    FeatureType.STRIDE_LENGTH_M1: "m",
    FeatureType.STRIDE_LENGTH_M1_ASYMMETRY: "ratio",
    FeatureType.GAIT_SPEED_M1: "m/s",
    FeatureType.GAIT_SPEED_M1_ASYMMETRY: "ratio",
    FeatureType.INTRA_STEP_COVARIANCE_V: "m2/s4",
    FeatureType.INTRA_STRIDE_COVARIANCE_V: "m2/s4",
    FeatureType.HARMONIC_RATIO_V: "ratio",
    FeatureType.STRIDE_SPARC: "a.u.",
}


def _build_epoch_unit_map() -> Dict[FeatureType, str]:
    mapping: Dict[FeatureType, str] = {}
    for feature_type in FeatureType.get_epoch_feature_types():
        value = feature_type.value
        if "dominant frequency" in value:
            mapping[feature_type] = "Hz"
        elif "spectral entropy" in value:
            mapping[feature_type] = "a.u."
        elif "bandpower" in value:
            mapping[feature_type] = "m2/s4"
        elif "jerk rms" in value:
            mapping[feature_type] = "m/s3"
        elif "zcr" in value:
            mapping[feature_type] = "1/s"
        else:
            mapping[feature_type] = "m/s2"
    return mapping


EPOCH_DEFAULT_UNIT_MAP = _build_epoch_unit_map()


def get_feature_unit(feature_type: FeatureType) -> str:
    if feature_type in STRIDE_DEFAULT_UNIT_MAP:
        return STRIDE_DEFAULT_UNIT_MAP[feature_type]
    if feature_type in EPOCH_DEFAULT_UNIT_MAP:
        return EPOCH_DEFAULT_UNIT_MAP[feature_type]
    return "a.u."
