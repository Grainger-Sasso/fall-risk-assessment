from enum import Enum


class RawFeatureFields(Enum):
    """Fields used in raw feature files"""

    RAW_FEATURE = "raw_feature"
    FEATURES = "features"
    EPOCH_STARTS = "epoch_starts"
    EPOCH_ENDS = "epoch_ends"
    FEATURE_NAMES = "feature_names"
    RAW_FEATURE_IDENTIFIER = "raw_feature_identifier"
    IMU_DATA_IDENTIFIER = "imu_data_identifier"
    USER_DATA_IDENTIFIER = "user_data_identifier"
    START_TIME = "start_time"
    EPOCH_LEN = "epoch_len"
