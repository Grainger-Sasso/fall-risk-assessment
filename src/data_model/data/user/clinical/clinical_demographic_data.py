from dataclasses import dataclass
from enum import Enum
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
    age: int
    sex: Sex
    weight: Kilogram
    height: Meter
