from enum import Enum


class AggregateFeatureFields(Enum):
    """Fields used in aggregate feature files"""

    AGGREGATE_FEATURE = "aggregate_feature"
    FEATURES = "features"
    FEATURE_NAMES = "feature_names"
    DESCRIPTIVE_STATISTIC_NAMES = "descriptive_statistic_names"
    AGGREGATE_FEATURE_IDENTIFIER = "aggregate_feature_identifier"
    RAW_FEATURE_IDENTIFIER = "raw_feature_identifier"
    IMU_DATA_IDENTIFIER = "imu_data_identifier"
    USER_DATA_IDENTIFIER = "user_data_identifier"
