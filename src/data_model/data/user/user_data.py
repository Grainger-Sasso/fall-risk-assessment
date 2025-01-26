from dataclasses import dataclass
from typing import List

from src.data_model.data.user.clinical.clinical_assessment import ClinicalAssessment
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
)


@dataclass
class UserData:
    user_id: str
    clinical_demographic_data: ClinicalDemographicData
    clinical_assessments: List[ClinicalAssessment]
