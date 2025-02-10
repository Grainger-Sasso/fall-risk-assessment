from pathlib import Path
from typing import Dict

from src.identifiers.user.user_identifier import UserIdentifier
from src.database_manager.registries.registry import Registry


class UserDataRegistry(Registry):
    def __init__(self, registry: Dict[UserIdentifier:Path]):
        super().__init__(registry)
