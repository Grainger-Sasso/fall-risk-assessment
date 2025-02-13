from typing import List

from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.model_fields.feature_set.feature_set_fields import FeatureSetFields
from src.data_model.feature_set.feature_set import FeatureSet
from src.data_model.feature_set.feature_set_entry import FeatureSetEntry
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)


class FeatureSetBuilder(ModelBuilder):
    """Builds FeatureSet from input CSV file"""

    version = "1.0"

    def build(self, input_file: CSVFile, feature_set_name: str) -> FeatureSet:
        raw_feature_identifiers = [
            RawFeatureIdentifier(id)
            for id in input_file.data[FeatureSetFields.RAW_FEATURE_IDENTIFIER]
        ]
        aggregate_feature_identifiers = [
            AggregateFeatureIdentifier(id)
            for id in input_file.data[FeatureSetFields.AGGREGATE_FEATURE_IDENTIFIER]
        ]
        entries: List[FeatureSetEntry] = []
        for raw_feature_id, aggregate_feature_id in zip(
            raw_feature_identifiers, aggregate_feature_identifiers
        ):
            entries.append(
                FeatureSetEntry(
                    raw_feature_identifier=raw_feature_id,
                    aggregate_feature_identifier=aggregate_feature_id,
                )
            )
        return FeatureSet(name=feature_set_name, entries=entries)
