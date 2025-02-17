from typing import Any, Dict, List

import numpy as np

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.hdf5.hdf5_group import HDF5Group
from src.data_io.model_fields.features.aggregate.aggregate_feature_fields import (
    AggregateFeatureFields,
)
from src.data_model.features.aggregate.aggregate_feature import AggregateFeature
from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.aggregate.descriptive_statistic import DescriptiveStatistic
from src.data_model.features.aggregate.metadata.aggregate_feature_set_entry_metadata import (
    AggregateFeatureSetEntryMetadata,
)
from src.data_types.descriptive_statistics.descriptive_statistic_type import (
    DescriptiveStatisticType,
)
from src.data_types.feature.raw_feature_type import RawFeatureType
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class AggregateFeatureSetEntryBuilder(ModelBuilder):
    """Builds aggregate feature set entries from HDF5 file format.

    This builder handles conversion of HDF5 file data into aggregate feature models,
    including feature statistics and metadata.

    Attributes:
        version (str): Version identifier for the builder
        stat_name_to_descriptive_stat_type_map (Dict[str, DescriptiveStatisticType]):
            Maps statistic names to types
        feature_name_to_raw_feature_type_map (Dict[str, RawFeatureType]):
            Maps feature names to types
    """

    version: str = "1.0"

    def __init__(self):
        """TODO: fill in this mapping with feature types and names"""
        """Initialize the builder with mapping dictionaries.
        
        Note:
            Mappings need to be populated with actual feature types and names.
        """
        super().__init__()
        self.stat_name_to_descriptive_stat_type_map: Dict[
            str, DescriptiveStatisticType
        ] = {"placeholder": DescriptiveStatisticType.PLACEHOLDER}
        self.feature_name_to_raw_feature_type_map: Dict[str, RawFeatureType] = {
            "placeholder": RawFeatureType.PLACEHOLDER
        }

    def build(self, input_file: HDF5Group) -> AggregateFeatureSetEntry:
        """Build aggregate feature set entry from HDF5 group.

        Args:
            input_file (HDF5Group): Source HDF5 file data

        Returns:
            AggregateFeatureSetEntry: Constructed feature set entry

        Raises:
            ValueError: If required data is missing or invalid
        """
        if not input_file:
            raise ValueError("Input file data is required")

        # Build metadata
        input_file_attributes: Dict[str:Any] = input_file.attributes
        aggregate_feature_set_entry_metadata: AggregateFeatureSetEntryMetadata = (
            self.__build_aggregate_feature_set_entry_metadata(input_file_attributes)
        )
        # Build aggregate feature list
        aggregate_feature_list: List[AggregateFeature] = (
            self.__build_aggregate_feature_list(input_file)
        )
        return AggregateFeatureSetEntry(
            aggregate_feature_list, aggregate_feature_set_entry_metadata
        )

    def __build_aggregate_feature_list(
        self, input_file: HDF5Group
    ) -> List[AggregateFeature]:
        """Build list of aggregate features from HDF5 group.

        Args:
            input_file (HDF5Group): Source HDF5 file data

        Returns:
            List[AggregateFeature]: List of constructed features

        Raises:
            ValueError: If required data is missing or invalid
        """
        try:
            features: np.ndarray = np.array(
                input_file.get_item_by_name(AggregateFeatureFields.FEATURES).data
            )
            feature_row_indices: np.ndarray = np.array(
                input_file.get_item_by_name(AggregateFeatureFields.FEATURE_NAMES).data
            )
            feature_col_names: np.ndarray = np.array(
                input_file.get_item_by_name(
                    AggregateFeatureFields.DESCRIPTIVE_STATISTIC_NAMES
                ).data
            )
        except ValueError as e:
            raise ValueError(f"Missing required feature data: {e}")

        if len(features) == 0:
            raise ValueError("Empty features array")

        aggregate_feature_list: List[AggregateFeature] = []
        for row_index, feature_name in enumerate(feature_row_indices):
            if feature_name not in self.feature_name_to_raw_feature_type_map:
                raise ValueError(f"Unknown feature type: {feature_name}")

            raw_feature_type: RawFeatureType = (
                self.feature_name_to_raw_feature_type_map[feature_name]
            )
            aggregate_feature_list.append(
                self.__build_aggregate_feature(
                    features, raw_feature_type, row_index, feature_col_names
                )
            )
        return aggregate_feature_list

    def __build_aggregate_feature(
        self,
        features: np.ndarray,
        raw_feature_type: RawFeatureType,
        row_index: int,
        feature_col_names: np.ndarray,
    ) -> AggregateFeature:
        """Build single aggregate feature from feature data.

        Args:
            features (np.ndarray): Array of feature values
            raw_feature_type (RawFeatureType): Type of the feature
            row_index (int): Row index in features array
            feature_col_names (np.ndarray): Names of feature columns

        Returns:
            AggregateFeature: Constructed aggregate feature

        Raises:
            ValueError: If statistic mapping is invalid
        """
        descriptive_statistic_list: List[DescriptiveStatistic] = []
        for col_index, stat_name in enumerate(feature_col_names):
            if stat_name not in self.stat_name_to_descriptive_stat_type_map:
                raise ValueError(f"Unknown statistic type: {stat_name}")

            statistic_type: DescriptiveStatisticType = (
                self.stat_name_to_descriptive_stat_type_map(stat_name)
            )
            statistic_value: float = features[row_index][col_index]
            descriptive_statistic_list.append(
                DescriptiveStatistic(statistic_type, statistic_value)
            )
        return AggregateFeature(descriptive_statistic_list, raw_feature_type)

    def __build_aggregate_feature_set_entry_metadata(
        self, input_file_attributes: Dict[str:Any]
    ) -> AggregateFeatureSetEntryMetadata:
        """Build metadata for aggregate feature set entry.

        Args:
            input_file_attributes (Dict[str, Any]): Source file attributes

        Returns:
            AggregateFeatureSetEntryMetadata: Constructed metadata

        Raises:
            ValueError: If required metadata fields are missing
        """
        required_fields = [
            AggregateFeatureFields.AGGREGATE_FEATURE_IDENTIFIER,
            AggregateFeatureFields.RAW_FEATURE_IDENTIFIER,
            AggregateFeatureFields.USER_DATA_IDENTIFIER,
            AggregateFeatureFields.IMU_DATA_IDENTIFIER,
        ]

        for field in required_fields:
            if field.value not in input_file_attributes:
                raise ValueError(f"Missing required metadata field: {field.value}")

        aggregate_feature_identifier: AggregateFeatureIdentifier = (
            AggregateFeatureIdentifier(
                input_file_attributes[
                    AggregateFeatureFields.AGGREGATE_FEATURE_IDENTIFIER
                ]
            )
        )
        raw_feature_identifier: RawFeatureIdentifier = RawFeatureIdentifier(
            input_file_attributes[AggregateFeatureFields.RAW_FEATURE_IDENTIFIER]
        )
        user_identifier: UserIdentifier = UserIdentifier(
            input_file_attributes[AggregateFeatureFields.USER_DATA_IDENTIFIER]
        )
        imu_data_identifier: IMUDataIdentifier = IMUDataIdentifier(
            input_file_attributes[AggregateFeatureFields.IMU_DATA_IDENTIFIER]
        )
        return AggregateFeatureSetEntryMetadata(
            aggregate_feature_identifier,
            raw_feature_identifier,
            user_identifier,
            imu_data_identifier,
        )
