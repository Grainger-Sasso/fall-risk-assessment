from typing import Dict
from pathlib import Path
from src.data_io.builders.model_builders.registries.registry_builder import (
    RegistryBuilder,
)
from src.data_io.formats.csv.csv_file import CSVFile
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.database_manager.registries.imu.imu_data_registry import IMUDataRegistry


class IMUDataRegistryBuilder(RegistryBuilder):
    def build(self, input_file: CSVFile) -> IMUDataRegistry:
        registry: Dict[str, Path] = self.build_registry(input_file)
        return IMUDataRegistry(
            {IMUDataIdentifier(id): path for id, path in registry.items()}
        )
