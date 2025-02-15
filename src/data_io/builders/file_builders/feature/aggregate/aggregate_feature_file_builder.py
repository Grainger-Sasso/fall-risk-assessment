from typing import Any, Dict, Tuple

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
    """File builder for aggregate feature set entries"""

    def build(self, data: AggregateFeatureSetEntry) -> HDF5Group:
        aggregate_feature_group = HDF5Group()
        aggregate_feature_group.name = AggregateFeatureFields.AGGREGATE_FEATURE.value
        # Build aggregate feature group items
        aggregate_feature_group.items = [
            item for item in self.__build_aggregate_feature_group_items(data)
        ]
        # Build aggreate feature group attributes (metadata)
        aggregate_feature_group.attributes = (
            self.__build_aggregate_feature_group_attributes(data.metadata)
        )
        return aggregate_feature_group

    def __build_aggregate_feature_group_items(
        self, data: AggregateFeatureSetEntry
    ) -> Tuple[HDF5Dataset]:
        feature_names = []
        aggregate_feature_data = []
        for aggregate_feature in data.aggregate_features:
            feature_names.append[aggregate_feature.feature_type.value]
            aggregate_feature = [
                stat.value for stat in aggregate_feature.descriptive_statistics
            ]
            aggregate_feature_data.append(aggregate_feature)
        stat_names = [
            stat_name.value
            for stat_name in data.aggregate_features[0]._statistics_map.keys
        ]
        # Build feature dataset
        feature_dataset = HDF5Dataset()
        feature_dataset.name = AggregateFeatureFields.FEATURES.value
        feature_dataset.attributes = {}
        feature_dataset.data = aggregate_feature_data
        # Build feature name dataset
        feature_names_dataset = HDF5Dataset()
        feature_names_dataset.name = AggregateFeatureFields.FEATURE_NAMES.value
        feature_names_dataset.attributes = {}
        feature_names_dataset.data = feature_names
        # Build stat name dataset
        stat_names_dataset = HDF5Dataset()
        stat_names_dataset.name = (
            AggregateFeatureFields.DESCRIPTIVE_STATISTIC_NAMES.value
        )
        stat_names_dataset.attributes = {}
        stat_names_dataset.data = stat_names
        return tuple(feature_dataset, feature_names_dataset, stat_names_dataset)

    def __build_aggregate_feature_group_attributes(
        self, metadata: AggregateFeatureSetEntryMetadata
    ) -> Dict[str, Any]:
        aggregate_feature_id: str = metadata.aggregate_feature_identifier.value
        raw_feature_id: str = metadata.imu_data_identifier.value
        user_id: str = metadata.user_identifier.value
        imu_data_id: str = metadata.imu_data_identifier.value
        return {
            AggregateFeatureFields.AGGREGATE_FEATURE_IDENTIFIER.value: aggregate_feature_id,
            AggregateFeatureFields.RAW_FEATURE_IDENTIFIER.value: raw_feature_id,
            AggregateFeatureFields.USER_DATA_IDENTIFIER.value: user_id,
            AggregateFeatureFields.IMU_DATA_IDENTIFIER.value: imu_data_id,
        }
