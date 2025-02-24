from typing import List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.feature_set.feature_set_fields import FeatureSetFields
from src.data_model.feature_set.feature_set import FeatureSet
from src.data_model.feature_set.feature_set_entry import FeatureSetEntry
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier


class FeatureSetBuilder(ModelBuilder):
    """Builds FeatureSet from input CSV file"""

    version = "1.0"

    def build(self, input_file: CSVFile, feature_set_name: str) -> FeatureSet:
        """Build FeatureSet from CSV file.

        Args:
            input_file (CSVFile): Input CSV file containing feature set entries
            feature_set_name (str): Name of the feature set

        Returns:
            FeatureSet: Built feature set object

        Raises:
            ValueError: If input file is invalid
        """
        if not isinstance(input_file, CSVFile) or not feature_set_name:
            raise ValueError("Invalid feature set CSV file or name")

        raw_feature_identifiers = [
            RawFeatureIdentifier(id)
            for id in input_file.data[FeatureSetFields.RAW_FEATURE_IDENTIFIER.value]
        ]
        aggregate_feature_identifiers = [
            AggregateFeatureIdentifier(id)
            for id in input_file.data[
                FeatureSetFields.AGGREGATE_FEATURE_IDENTIFIER.value
            ]
        ]
        entries = [
            FeatureSetEntry(
                raw_feature_identifier=raw_id,
                aggregate_feature_identifier=agg_id,
            )
            for raw_id, agg_id in zip(
                raw_feature_identifiers, aggregate_feature_identifiers
            )
        ]
        return FeatureSet(name=feature_set_name, entries=entries)
