from dataclasses import dataclass
from typing import List

from src.data_model.assessment_data import AssessmentData
from src.data_model.data.user.clinical.clinical_assessment import ClinicalAssessment
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
)
from src.identifiers.identifier import Identifier
from src.identifiers.user.user_identifier import UserIdentifier


@dataclass
class UserData(AssessmentData):
    user_identifier: UserIdentifier
    clinical_demographic_data: ClinicalDemographicData
    clinical_assessments: List[ClinicalAssessment]

    def get_data_id(self) -> Identifier:
        return self.user_identifier
