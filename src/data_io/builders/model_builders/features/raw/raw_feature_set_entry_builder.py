from typing import Any, Dict, List

import numpy as np  # type: ignore

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.raw.raw_feature_fields import RawFeatureFields
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_epoch_features import RawEpochFeatures
from src.data_model.features.raw.raw_feature import RawFeature
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class RawFeatureSetEntryBuilder(ModelBuilder):
    """Builds raw feature set entries from HDF5 file format.

    This builder handles conversion of HDF5 file data into raw feature models,
    including feature values, epochs, and metadata.

    Attributes:
        version (str): Version identifier for the builder
    """

    version: str = "1.0"

    def __init__(self):
        """TODO: fill in this mapping with feature types and names"""
        """Initialize the builder with feature type mapping.
        
        Note:
            Mapping needs to be populated with actual feature types and names.
        """
        super().__init__()

    def build(self, input_file: HDF5Group) -> RawFeatureSetEntry:
        """Build raw feature set entry from HDF5 group.

        Args:
            input_file (HDF5Group): Source HDF5 file data

        Returns:
            RawFeatureSetEntry: Constructed feature set entry

        Raises:
            ValueError: If required data is missing or invalid
        """
        if not isinstance(input_file, HDF5Group):
            raise ValueError("File must raw feature data from HDF5")
        return self.__build_raw_feature_set_entry(input_file)

    def __build_raw_feature_set_entry(
        self, input_file: HDF5Group
    ) -> RawFeatureSetEntry:
        # Build metadata
        input_file_attributes: Dict[str:Any] = input_file.attributes
        raw_feature_set_entry_metadata: RawFeatureSetEntryMetadata = (
            self.__build_raw_feature_set_entry_metadata(input_file_attributes)
        )
        # Build raw epoch feature list
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
        """Build list of raw epoch features from HDF5 group.

        Args:
            input_file (HDF5Group): Source HDF5 file data
            epoch_len (float): Length of each epoch in seconds

        Returns:
            List[RawEpochFeatures]: List of constructed epoch features

        Raises:
            ValueError: If required data is missing or invalid
        """
        if epoch_len <= 0:
            raise ValueError("Epoch length must be positive")

        try:
            features: np.ndarray = np.array(
                input_file.get_item_by_name(RawFeatureFields.FEATURES.value).data
            )
            epoch_starts: np.ndarray = np.array(
                input_file.get_item_by_name(RawFeatureFields.EPOCH_STARTS.value).data
            )
            epoch_ends: np.ndarray = np.array(
                input_file.get_item_by_name(RawFeatureFields.EPOCH_ENDS.value).data
            )
            feature_names: np.ndarray = np.array(
                input_file.get_item_by_name(RawFeatureFields.FEATURE_NAMES.value).data
            )
        except ValueError as e:
            raise ValueError(f"Missing required feature data: {e}")

        if len(features) == 0:
            raise ValueError("Empty features array")
        if len(epoch_starts) == 0:
            raise ValueError("Empty epoch start time array")
        if len(epoch_ends) == 0:
            raise ValueError("Empty epoch end time array")
        if len(feature_names) == 0:
            raise ValueError("Empty feature names array")
        if features.shape[0] != len(epoch_starts):
            raise ValueError(
                "Number of features does not match number of epoch start times"
            )
        if features.shape[0] != len(epoch_ends):
            raise ValueError(
                "Number of features does not match number of epoch end times"
            )
        if features.shape[1] != len(feature_names):
            raise ValueError(
                "Number of features does not match number of feature names"
            )

        raw_epoch_feature_list: List[RawEpochFeatures] = []
        for row_index in range(len(epoch_starts)):
            raw_epoch_feature_list.append(
                self.__build_raw_epoch_features(
                    features,
                    feature_names,
                    row_index,
                    epoch_starts[row_index],
                    epoch_ends[row_index],
                )
            )
        return raw_epoch_feature_list

    def __build_raw_epoch_features(
        self,
        features: np.ndarray,
        feature_col_names: np.ndarray,
        row_index: int,
        epoch_start: float,
        epoch_end: float,
    ) -> RawEpochFeatures:
        """Build raw epoch features from feature data.

        Args:
            features (np.ndarray): 2D array of feature values
            feature_col_names (np.ndarray): Array of feature names
            row_index (int): Index of current epoch in features array
            epoch (float): Start time of the epoch
            epoch_end (float): End time of the epoch

        Returns:
            RawEpochFeatures: Constructed epoch features

        Raises:
            ValueError: If feature type mapping is invalid or data is inconsistent
        """
        if row_index < 0 or row_index >= features.shape[0]:
            raise ValueError(f"Invalid row index: {row_index}")

        # Build raw feature list
        raw_feature_list: List[RawFeature] = []
        for col_index, feature_name in enumerate(feature_col_names):
            try:
                raw_feature_type: RawFeatureType = RawFeatureType(feature_name)
            except ValueError:
                # Raise an error if the string is not a valid enum member
                raise ValueError(
                    f"'{feature_name}' is not a valid member of {RawFeatureType.__name__}"
                )

            feature_value: float = features[row_index][col_index]
            raw_feature_list.append(RawFeature(raw_feature_type, feature_value))

        if not raw_feature_list:
            raise ValueError("No features constructed for epoch")

        return RawEpochFeatures(raw_feature_list, epoch_start, epoch_end)

    def __build_raw_feature_set_entry_metadata(
        self, input_file_attributes: Dict[str, Any]
    ) -> RawFeatureSetEntryMetadata:
        """Build metadata for raw feature set entry.

        Args:
            input_file_attributes (Dict[str, Any]): Source file attributes

        Returns:
            RawFeatureSetEntryMetadata: Constructed metadata

        Raises:
            ValueError: If required metadata fields are missing
        """
        required_fields = [
            RawFeatureFields.RAW_FEATURE_IDENTIFIER,
            RawFeatureFields.USER_DATA_IDENTIFIER,
            RawFeatureFields.IMU_DATA_IDENTIFIER,
            RawFeatureFields.START_TIME,
            RawFeatureFields.EPOCH_LEN,
        ]

        for field in required_fields:
            if field.value not in input_file_attributes:
                raise ValueError(f"Missing required metadata field: {field.value}")

        raw_feature_identifier: RawFeatureIdentifier = RawFeatureIdentifier(
            input_file_attributes[RawFeatureFields.RAW_FEATURE_IDENTIFIER.value]
        )
        user_identifier: UserIdentifier = UserIdentifier(
            input_file_attributes[RawFeatureFields.USER_DATA_IDENTIFIER.value]
        )
        imu_data_identifier: IMUDataIdentifier = IMUDataIdentifier(
            input_file_attributes[RawFeatureFields.IMU_DATA_IDENTIFIER.value]
        )
        start_time: float = input_file_attributes[RawFeatureFields.START_TIME.value]
        epoch_length: float = input_file_attributes[RawFeatureFields.EPOCH_LEN.value]

        if epoch_length <= 0:
            raise ValueError("Epoch length must be positive")

        return RawFeatureSetEntryMetadata(
            raw_feature_identifier,
            user_identifier,
            imu_data_identifier,
            start_time,
            epoch_length,
        )
