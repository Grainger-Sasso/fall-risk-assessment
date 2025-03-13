from dataclasses import dataclass
from enum import Enum

from src.identifiers.user.clinical_identifier import ClinicalIdentifier
from src.util.mechanics.units.si.kilogram import Kilogram
from src.util.mechanics.units.si.meter import Meter


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


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
