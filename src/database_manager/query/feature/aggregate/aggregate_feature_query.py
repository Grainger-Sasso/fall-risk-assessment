from typing import Dict, List, Tuple

from src.data_model.features.aggregate.aggregate_feature_set_entry import (
    AggregateFeatureSetEntry,
)
from src.data_model.features.raw.raw_feature_set_entry import RawFeatureSetEntry
from src.database_manager.query.query_interface import RelatedDataQueryInterface
from src.identifiers.feature.aggregate_feature_identifier import (
    AggregateFeatureIdentifier,
)
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier


class AggregateFeatureQuery(
    RelatedDataQueryInterface[AggregateFeatureSetEntry, AggregateFeatureIdentifier]
):
    """Interface for querying aggregate feature data"""

    def __init__(
        self,
        *args,
        raw_feature_query: RelatedDataQueryInterface[
            RawFeatureSetEntry, RawFeatureIdentifier
        ],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._raw_feature_query = raw_feature_query

    def get_aggregate_feature(
        self, feature_id: AggregateFeatureIdentifier
    ) -> AggregateFeatureSetEntry:
        """Get aggregate feature by ID

        Args:
            feature_id: Aggregate feature identifier

        Returns:
            Aggregate feature set entry

        Raises:
            KeyError: If feature not found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data(feature_id)

    def get_all_aggregate_features(
        self,
    ) -> Dict[AggregateFeatureIdentifier, AggregateFeatureSetEntry]:
        """Get all aggregate features

        Returns:
            Dictionary mapping feature IDs to feature objects

        Raises:
            FileNotFoundError: If any data file not found
            ImportError: If any data import fails
        """
        return self.get_all_data()

    def get_raw_feature_aggregates(
        self, raw_id: RawFeatureIdentifier
    ) -> List[AggregateFeatureSetEntry]:
        """Get all aggregate features derived from a raw feature

        Args:
            raw_id: Raw feature identifier

        Returns:
            List of aggregate features derived from the raw feature

        Raises:
            KeyError: If no aggregates found for raw feature
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data_for_target(raw_id)

    def get_feature_chain(
        self, agg_feature_id: AggregateFeatureIdentifier
    ) -> Tuple[AggregateFeatureSetEntry, RawFeatureSetEntry]:
        """Get aggregate feature and its source raw feature

        Args:
            agg_feature_id: Aggregate feature identifier

        Returns:
            Tuple of (aggregate feature, raw feature)

        Raises:
            KeyError: If feature or source not found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        # Get aggregate feature
        agg_feature = self.get_data(agg_feature_id)

        # Get corresponding raw feature
        raw_id = self._mapping_manager.get_target_id(agg_feature_id)
        raw_feature = self._raw_feature_query.get_data(raw_id)

        return agg_feature, raw_feature 