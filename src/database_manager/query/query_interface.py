from pathlib import Path
from typing import Dict, Generic, List, TypeVar

from src.data_io.import_export.importers.importer import Importer
from src.database_manager.data_access.data_loader import DataLoader
from src.database_manager.data_access.mapping_manager import MappingManager
from src.database_manager.data_access.registry_manager import RegistryManager
from src.identifiers.identifier import Identifier

T = TypeVar("T")  # Type of data model
S = TypeVar("S", bound=Identifier)  # Source identifier type
U = TypeVar("U", bound=Identifier)  # Target identifier type


class QueryInterface(Generic[T, S]):
    """Interface for querying data from the database"""

    def __init__(
        self,
        registry_manager: RegistryManager[S],
        data_loader: DataLoader[T],
    ):
        self._registry_manager = registry_manager
        self._data_loader = data_loader

    def get_data(self, identifier: S) -> T:
        """Get data for a given identifier

        Args:
            identifier: Identifier for the data

        Returns:
            Data model object

        Raises:
            KeyError: If identifier not found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        path = self._registry_manager.get_path(identifier)
        return self._data_loader.load(path)

    def get_all_data(self) -> Dict[S, T]:
        """Get all data in the registry

        Returns:
            Dictionary mapping identifiers to data objects

        Raises:
            FileNotFoundError: If any data file not found
            ImportError: If any data import fails
        """
        results = {}
        for id, path in self._registry_manager.get_all_paths().items():
            results[id] = self._data_loader.load(path)
        return results


class RelatedDataQueryInterface(QueryInterface[T, S]):
    """Interface for querying data with relationships to other data types"""

    def __init__(
        self,
        registry_manager: RegistryManager[S],
        data_loader: DataLoader[T],
        mapping_manager: MappingManager[S, U],
    ):
        super().__init__(registry_manager, data_loader)
        self._mapping_manager = mapping_manager

    def get_data_for_target(self, target_id: U) -> List[T]:
        """Get all data related to a target identifier

        Args:
            target_id: Target identifier to find related data for

        Returns:
            List of related data objects

        Raises:
            KeyError: If no related data found
            FileNotFoundError: If data file not found
            ImportError: If data import fails
        """
        # Find all source IDs that map to this target
        source_ids = [
            source_id
            for source_id, tid in self._mapping_manager._mapping.items()
            if tid == target_id
        ]
        
        if not source_ids:
            raise KeyError(f"No data found related to target ID: {target_id}")

        # Load data for each source ID
        return [self.get_data(source_id) for source_id in source_ids] 