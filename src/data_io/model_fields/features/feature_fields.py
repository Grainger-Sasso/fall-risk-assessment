from enum import Enum


class FeatureFields(Enum):
    """Fields used in feature files"""

    RECORD_FEATURES = "record_features"
    EPOCH_FEATURES = "epoch_features"
    STRIDE_FEATURES = "stride_features"
    FEATURES = "features"
    BOUT_STARTS = "bout_starts"
    BOUT_ENDS = "bout_ends"
    FEATURE_NAMES = "feature_names"
    SAMPLE_STARTS = "sample_starts"
    SAMPLE_ENDS = "sample_ends"
    UNITS = "units"
    FEATURE_IDENTIFIER = "feature_identifier"
    IMU_DATA_IDENTIFIER = "imu_data_identifier"
    USER_DATA_IDENTIFIER = "user_data_identifier"
    VERSION = "version"
    EXTRACTION_BACKEND = "extraction_backend"
    STRIDE_FEATURE_CATALOG = "stride_feature_catalog"
    EXTRACTION_PROFILE = "extraction_profile"
    EXTRACTION_LIBRARY_VERSION = "extraction_library_version"
    EXTRACTED_AT_UTC = "extracted_at_utc"
