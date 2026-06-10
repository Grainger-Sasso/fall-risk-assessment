from src.data_types.feature.mobgap_feature_type import MobgapFeatureType


def get_mobgap_feature_unit(feature_type: MobgapFeatureType) -> str:
    units = {
        MobgapFeatureType.IC_SAMPLE: "samples",
        MobgapFeatureType.FOOT: "label",
        MobgapFeatureType.STEP_TIME_S: "s",
        MobgapFeatureType.STRIDE_TIME_S: "s",
        MobgapFeatureType.CADENCE_SPM: "steps/min",
        MobgapFeatureType.STRIDE_LENGTH_M: "m",
        MobgapFeatureType.WALKING_SPEED_MPS: "m/s",
    }
    return units[feature_type]
