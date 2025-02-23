from typing import Any, Dict, List

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.raw.raw_feature_fields import RawFeatureFields
from src.data_model.features.raw.metadata.raw_feature_set_entry_metadata import (
    RawFeatureSetEntryMetadata,
)
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry


class RawFeatureSetEntryFileBuilder(FileBuilder):
    """Builds HDF5 file format from raw feature set entries.

    This builder handles conversion of raw feature data into HDF5 format,
    including feature values, epochs, and metadata.

    Attributes:
        version (str): Version identifier for the builder
    """

    version: str = "1.0"

    def build(self, data: RawFeatureSetEntry) -> HDF5Group:
        """Build HDF5 group from aggregate feature set entry.

        Args:
            data (AggregateFeatureSetEntry): The feature data to convert

        Returns:
            HDF5Group: Root group containing all feature data
        """
        if not isinstance(data, RawFeatureSetEntry):
            raise ValueError("Data must be raw feature set entry")
        return self.__build_raw_feature_group(data)

    def __build_raw_feature_group(self, data: RawFeatureSetEntry) -> HDF5Group:
        """Build HDF5 group from raw feature set entry.

        Args:
            data (RawFeatureSetEntry): The feature data to convert

        Returns:
            HDF5Group: Root group containing all feature data

        Raises:
            ValueError: If data validation fails
        """

        return HDF5Group(
            name=RawFeatureFields.RAW_FEATURE.value,
            items=self.__build_raw_feature_group_items(data),
            attributes=self.__build_raw_feature_group_attributes(data.metadata),
        )

    def __build_raw_feature_group_items(
        self, data: RawFeatureSetEntry
    ) -> List[HDF5Dataset]:
        """Build HDF5 datasets for feature data.

        Args:
            data (RawFeatureSetEntry): Source feature data

        Returns:
            Tuple[HDF5Dataset]: Feature, epoch, and feature names datasets

        Raises:
            ValueError: If no epoch features are present
        """
        if not data.raw_epoch_features:
            raise ValueError("No epoch features found in data")

        # Get epochs, feature names, and N-D matrix of feature data
        epochs = []
        feature_data = []
        for raw_epoch_feature in data.raw_epoch_features:
            epochs.append(raw_epoch_feature.epoch_start_time)
            epoch_data = [
                raw_feature.value for raw_feature in raw_epoch_feature.raw_features
            ]
            feature_data.append(epoch_data)
        # Feature names are assumed to be consistent across all epochs
        feature_names = [
            feature.feature_type.value
            for feature in data.raw_epoch_features[0].raw_features
        ]

        # Build feature dataset
        feature_dataset: HDF5Dataset = HDF5Dataset(
            name=RawFeatureFields.FEATURES.value, data=feature_data, attributes={}
        )

        # Build feature epoch dataset
        feature_epoch_dataset: HDF5Dataset = HDF5Dataset(
            name=RawFeatureFields.FEATURE_EPOCHS.value, data=epochs, attributes={}
        )

        # Build feature name dataset
        feature_names_dataset: HDF5Dataset = HDF5Dataset(
            name=RawFeatureFields.FEATURE_NAMES.value, data=feature_names, attributes={}
        )

        return [feature_dataset, feature_epoch_dataset, feature_names_dataset]

    def __build_raw_feature_group_attributes(
        self, metadata: RawFeatureSetEntryMetadata
    ) -> Dict[str, Any]:
        """Build metadata attributes for the feature group.

        Args:
            metadata (RawFeatureSetEntryMetadata): Source metadata

        Returns:
            Dict[str, Any]: Dictionary of metadata attributes

        Raises:
            ValueError: If required metadata fields are missing
        """
        if not metadata:
            raise ValueError("Feature metadata is required")

        raw_feature_id: str = metadata.raw_feature_identifier.value
        user_id: str = metadata.user_identifier.value
        imu_data_id: str = metadata.imu_data_identifier.value
        start_time: float = metadata.start_time
        epoch_len: float = metadata.epoch_length

        if epoch_len <= 0:
            raise ValueError("Epoch length must be positive")

        return {
            RawFeatureFields.RAW_FEATURE_IDENTIFIER.value: raw_feature_id,
            RawFeatureFields.USER_DATA_IDENTIFIER.value: user_id,
            RawFeatureFields.IMU_DATA_IDENTIFIER.value: imu_data_id,
            RawFeatureFields.START_TIME.value: start_time,
            RawFeatureFields.EPOCH_LEN.value: epoch_len,
        }
