from pathlib import Path
from typing import Dict

from src.data_model.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.data_model.registries.registry import Registry


class IMUDataRegistry(Registry):
    def __init__(self, registry: Dict[IMUDataIdentifier:Path]):
        super().__init__(registry)
