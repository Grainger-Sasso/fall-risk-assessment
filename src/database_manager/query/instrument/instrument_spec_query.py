from typing import Dict

from src.data_model.instrument.instrument_specification import InstrumentSpecification
from src.database_manager.query.query_interface import QueryInterface
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)


class InstrumentSpecificationQuery(
    QueryInterface[InstrumentSpecification, InstrumentSpecificationIdentifier]
):
    """Interface for querying instrument specifications"""

    def get_instrument_spec(
        self, spec_id: InstrumentSpecificationIdentifier
    ) -> InstrumentSpecification:
        """Get instrument specification by ID

        Args:
            spec_id: Instrument specification identifier

        Returns:
            Instrument specification object

        Raises:
            KeyError: If specification not found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        return self.get_data(spec_id)

    def get_all_instrument_specs(
        self,
    ) -> Dict[InstrumentSpecificationIdentifier, InstrumentSpecification]:
        """Get all instrument specifications

        Returns:
            Dictionary mapping specification IDs to specification objects

        Raises:
            FileNotFoundError: If any data file not found
            ImportError: If any data import fails
        """
        return self.get_all_data() 