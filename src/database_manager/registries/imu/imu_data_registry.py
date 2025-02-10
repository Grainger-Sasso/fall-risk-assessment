from pathlib import Path
from typing import Dict

from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.database_manager.registries.registry import Registry


class IMUDataRegistry(Registry):
    def __init__(self, registry: Dict[IMUDataIdentifier:Path]):
        super().__init__(registry)
