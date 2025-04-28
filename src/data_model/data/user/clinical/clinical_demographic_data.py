from dataclasses import dataclass
from enum import Enum
from typing import Union

from src.identifiers.user.clinical_identifier import ClinicalIdentifier
from src.util.mechanics.units.si.kilogram import Kilogram
from src.util.mechanics.units.si.meter import Meter


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


class FallerStatus(Enum):
    FALLER = "faller"
    NON_FALLER = "non-faller"
    UNKNOWN = "unknown"

    @classmethod
    def from_bool(cls, is_faller: bool) -> "FallerStatus":
        """Convert a boolean to FallerStatus."""
        return cls.FALLER if is_faller else cls.NON_FALLER

    def to_bool(self) -> Union[bool, None]:
        """Convert FallerStatus to boolean. Returns None for UNKNOWN."""
        if self == self.UNKNOWN:
            return None
        return self == self.FALLER


@dataclass
class ClinicalDemographicData:
    """
    Stores the clinical demographic data for users.
    """

    name: str
    age: float
    sex: Sex
    weight: Kilogram
    height: Meter
    clinical_identifier: ClinicalIdentifier
    faller_status: FallerStatus

    @property
    def is_faller(self) -> Union[bool, None]:
        """Returns the boolean representation of faller status for ML training.
        Returns None if the status is unknown."""
        return self.faller_status.to_bool()
