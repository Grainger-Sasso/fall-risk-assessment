from typing import Any, Dict, Tuple

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
        """Build HDF5 group from raw feature set entry.

        Args:
            data (RawFeatureSetEntry): The feature data to convert

        Returns:
            HDF5Group: Root group containing all feature data

        Raises:
            ValueError: If data validation fails
        """
        if not data:
            raise ValueError("Raw feature data is required")

        raw_feature_group = HDF5Group()
        raw_feature_group.name = RawFeatureFields.RAW_FEATURE.value
        # Build raw feature group items
        raw_feature_group.items = [
            item for item in self.__build_raw_feature_group_items(data)
        ]
        # Build raw feature group attributes
        raw_feature_group.attributes = self.__build_raw_feature_group_attributes(
            data.metadata
        )
        return raw_feature_group

    def __build_raw_feature_group_items(
        self, data: RawFeatureSetEntry
    ) -> Tuple[HDF5Dataset]:
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
            name.value for name in data.raw_epoch_features[0]._raw_feature_map.keys()
        ]

        # Build feature dataset
        feature_dataset: HDF5Dataset = HDF5Dataset()
        feature_dataset.name = RawFeatureFields.FEATURES.value
        feature_dataset.attributes = {}
        feature_dataset.data = feature_data

        # Build feature epoch dataset
        feature_epoch_dataset: HDF5Dataset = HDF5Dataset()
        feature_epoch_dataset.name = RawFeatureFields.FEATURE_EPOCHS.value
        feature_epoch_dataset.attributes = {}
        feature_epoch_dataset.data = epochs

        # Build feature name dataset
        feature_names_dataset: HDF5Dataset = HDF5Dataset()
        feature_names_dataset.name = RawFeatureFields.FEATURE_NAMES.value
        feature_names_dataset.attributes = {}
        feature_names_dataset.data = feature_names

        return (feature_dataset, feature_epoch_dataset, feature_names_dataset)

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

        raw_feature_id: str = metadata.imu_data_identifier.value
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
