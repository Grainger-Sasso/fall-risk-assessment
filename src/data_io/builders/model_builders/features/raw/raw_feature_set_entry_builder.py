import numpy as np

from typing import List, Any, Dict
from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.raw.raw_feature_fields import RawFeatureFields
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_model.features.raw.raw_epoch_features import RawEpochFeatures
from src.data_model.features.raw.raw_feature import RawFeature
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


class RawFeatureSetEntryBuilder(ModelBuilder):
    """Raw feature set entry builder"""

    version = "1.0"

    def __init__(self):
        """TODO: fill in this mapping with feature types and names"""
        self.feature_name_to_raw_feature_type_map: Dict[str, RawFeatureType] = {
            "placeholder": RawFeatureType.PLACEHOLDER
        }

    def build(self, input_file: HDF5Group) -> RawFeatureSetEntry:
        # Build metadata
        input_file_attributes: Dict[str:Any] = input_file.attributes
        raw_feature_set_entry_metadata: RawFeatureSetEntryMetadata = (
            self.__build_raw_feature_set_entry_metadata(input_file_attributes)
        )
        # Build list of raw epoch features
        raw_epoch_feature_list: List[RawEpochFeatures] = (
            self.__build_raw_epoch_feature_list(
                input_file, raw_feature_set_entry_metadata.epoch_length
            )
        )
        return RawFeatureSetEntry(
            raw_epoch_feature_list, raw_feature_set_entry_metadata
        )

    def __build_raw_epoch_feature_list(
        self, input_file: HDF5Group, epoch_len: float
    ) -> List[RawEpochFeatures]:
        features: np.ndarray = np.array(
            input_file.get_item_by_name(RawFeatureFields.FEATURES).data
        )
        feature_row_indices: np.ndarray = np.array(
            input_file.get_item_by_name(RawFeatureFields.FEATURE_EPOCHS).data
        )
        feature_col_names: np.ndarray = np.array(
            input_file.get_item_by_name(RawFeatureFields.FEATURE_NAMES).data
        )
        # Build raw features2
        raw_epoch_feature_list: List[RawEpochFeatures] = []
        for row_index, epoch in enumerate(feature_row_indices):
            raw_epoch_feature_list.append(
                self.__build_raw_epoch_features(
                    features, feature_col_names, row_index, epoch, epoch_len
                )
            )
        return raw_epoch_feature_list

    def __build_raw_epoch_features(
        self,
        features: np.ndarray,
        feature_col_names: np.ndarray,
        row_index: int,
        epoch: float,
        epoch_len: float,
    ) -> RawEpochFeatures:
        # Build raw feature list
        raw_feature_list: List[RawFeature] = []
        for col_index, feature_name in enumerate(feature_col_names):
            feature_value: float = features[row_index][col_index]
            feature_type: RawFeatureType = self.feature_name_to_raw_feature_type_map[
                feature_name
            ]
            raw_feature_list.append(RawFeature(feature_type, feature_value))
        # Get epoch start and end time
        epoch_start_time, epoch_end_time = (epoch, epoch + epoch_len)
        return RawEpochFeatures(raw_feature_list, epoch_start_time, epoch_end_time)

    def __build_raw_feature_set_entry_metadata(
        self, input_file_attributes: Dict[str:Any]
    ) -> RawFeatureSetEntryMetadata:
        raw_feature_identifier: RawFeatureIdentifier = RawFeatureIdentifier(
            input_file_attributes[RawFeatureFields.RAW_FEATURE_IDENTIFIER]
        )
        user_identifier: UserIdentifier = UserIdentifier(
            input_file_attributes[RawFeatureFields.USER_DATA_IDENTIFIER]
        )
        imu_data_identifier: IMUDataIdentifier = IMUDataIdentifier(
            input_file_attributes[RawFeatureFields.IMU_DATA_IDENTIFIER]
        )
        start_time: float = input_file_attributes[RawFeatureFields.START_TIME]
        epoch_length: float = input_file_attributes[RawFeatureFields.EPOCH_LEN]
        return RawFeatureSetEntryMetadata(
            raw_feature_identifier,
            user_identifier,
            imu_data_identifier,
            start_time,
            epoch_length,
        )
