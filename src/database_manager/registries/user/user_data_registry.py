from pathlib import Path
from typing import Dict

from src.database_manager.registries.registry import Registry
from src.identifiers.user.user_identifier import UserIdentifier


class UserDataRegistry(Registry):
    def __init__(self, registry: Dict[UserIdentifier, Path]):
        super().__init__(registry)
