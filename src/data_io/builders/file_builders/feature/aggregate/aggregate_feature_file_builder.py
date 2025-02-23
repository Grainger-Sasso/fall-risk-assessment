from typing import Any, Dict, List

from src.data_io.builders.file_builders.file_builder import FileBuilder
from src.data_io.formats.hdf5.hdf5_dataset import HDF5Dataset
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.aggregate.aggregate_feature_fields import (
    AggregateFeatureFields,
)
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.aggregate.metadata.aggregate_feature_set_entry_metadata import (
    AggregateFeatureSetEntryMetadata,
)


class AggregateFeatureSetEntryFileBuilder(FileBuilder):
    """Builds HDF5 file format from aggregate feature set entries.

    This builder handles conversion of aggregate feature data into HDF5 format,
    including feature values, statistics, and metadata.

    Attributes:
        version (str): Version identifier for the builder
    """

    version: str = "1.0"

    def build(self, data: AggregateFeatureSetEntry) -> HDF5Group:
        """Build HDF5 group from aggregate feature set entry.

        Args:
            data (AggregateFeatureSetEntry): The feature data to convert

        Returns:
            HDF5Group: Root group containing all feature data
        """
        if not isinstance(data, AggregateFeatureSetEntry):
            raise ValueError("Data must be aggregate feature set entry")
        return self.__build_aggregate_feature_group(data)

    def __build_aggregate_feature_group(
        self, data: AggregateFeatureSetEntry
    ) -> HDF5Group:
        aggregate_feature_group_name = AggregateFeatureFields.AGGREGATE_FEATURE.value
        # Build aggregate feature group items
        aggregate_feature_group_items = self.__build_aggregate_feature_group_items(data)
        # Build aggreate feature group attributes (metadata)
        aggregate_feature_group_attributes = (
            self.__build_aggregate_feature_group_attributes(data.metadata)
        )
        return HDF5Group(
            name=aggregate_feature_group_name,
            items=aggregate_feature_group_items,
            attributes=aggregate_feature_group_attributes,
        )

    def __build_aggregate_feature_group_items(
        self, data: AggregateFeatureSetEntry
    ) -> List[HDF5Dataset]:
        """Build HDF5 datasets for feature data.

        Args:
            data (AggregateFeatureSetEntry): Source feature data

        Returns:
            Tuple[HDF5Dataset]: Feature, feature names, and statistic names datasets
        """
        feature_names = []
        aggregate_feature_data = []
        for aggregate_feature in data.aggregate_features:
            feature_names.append(aggregate_feature.feature_type.value)
            aggregate_feature_data.append(aggregate_feature.get_stat_values())
        stat_names = [
            stat.statistic_type.value
            for stat in data.aggregate_features[0].descriptive_statistics
        ]
        # Build feature dataset
        feature_dataset = HDF5Dataset(
            name=AggregateFeatureFields.FEATURES.value,
            data=aggregate_feature_data,
            attributes={},
        )
        # Build feature name dataset
        feature_names_dataset = HDF5Dataset(
            name=AggregateFeatureFields.FEATURE_NAMES.value,
            data=feature_names,
            attributes={},
        )
        # Build stat name dataset
        stat_names_dataset = HDF5Dataset(
            name=AggregateFeatureFields.DESCRIPTIVE_STATISTIC_NAMES.value,
            data=stat_names,
            attributes={},
        )
        return [feature_dataset, feature_names_dataset, stat_names_dataset]

    def __build_aggregate_feature_group_attributes(
        self, metadata: AggregateFeatureSetEntryMetadata
    ) -> Dict[str, Any]:
        """Build metadata attributes for the feature group.

        Args:
            metadata (AggregateFeatureSetEntryMetadata): Source metadata

        Returns:
            Dict[str, Any]: Dictionary of metadata attributes

        Raises:
            ValueError: If required metadata fields are missing
        """
        if not metadata:
            raise ValueError("Feature metadata is required")

        aggregate_feature_id: str = metadata.aggregate_feature_identifier.value
        raw_feature_id: str = metadata.raw_feature_identifier.value
        user_id: str = metadata.user_identifier.value
        imu_data_id: str = metadata.imu_data_identifier.value
        return {
            AggregateFeatureFields.AGGREGATE_FEATURE_IDENTIFIER.value: aggregate_feature_id,
            AggregateFeatureFields.RAW_FEATURE_IDENTIFIER.value: raw_feature_id,
            AggregateFeatureFields.USER_DATA_IDENTIFIER.value: user_id,
            AggregateFeatureFields.IMU_DATA_IDENTIFIER.value: imu_data_id,
        }
