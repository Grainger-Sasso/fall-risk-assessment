from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_model.data.user.user_data import UserData
from src.data_model.data.user.clinical.clinical_assessment import ClinicalAssessment
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
)
from src.data_model.data.user.clinical.clinical_demographic_data import Sex
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.identifiers.user.user_identifier import UserIdentifier
from src.data_io.model_fields.data.user.user_data_fields import UserUDataFields
from src.data_io.model_fields.data.user.clinical_demographic_data_fields import (
    ClinicalDemographicDataFields,
)
from src.util.mechanics.units.si.kilogram import Kilogram
from src.util.mechanics.units.si.meter import Meter


class UserDataBuilder(ModelBuilder):
    """User data builder"""

    version = "1.0"

    def build(
        self, input_file: JSONDictFile, clinical_demo_file: JSONDictFile
    ) -> UserData:
        user_identifier: UserIdentifier = UserIdentifier(
            input_file[UserUDataFields.USER_DATA_IDENTIFIER]
        )
        clinical_demo_data: ClinicalDemographicData = (
            self.__build_clinical_demographic_data(clinical_demo_file)
        )
        clinical_fall_risk_assessment: ClinicalAssessment = ClinicalAssessment()
        return UserData(
            user_identifier, clinical_demo_data, clinical_fall_risk_assessment
        )

    def __build_clinical_demographic_data(
        self, clinical_demo_file: JSONDictFile
    ) -> ClinicalDemographicData:
        name: str = clinical_demo_file[ClinicalDemographicDataFields.NAME][
            ClinicalDemographicDataFields.VALUE
        ]
        age: float = clinical_demo_file[ClinicalDemographicDataFields.AGE][
            ClinicalDemographicDataFields.VALUE
        ]
        sex: Sex = Sex(
            clinical_demo_file[ClinicalDemographicDataFields.SEX][
                ClinicalDemographicDataFields.VALUE
            ]
        )
        weight: Kilogram = Kilogram(
            clinical_demo_file[ClinicalDemographicDataFields.WEIGHT][
                ClinicalDemographicDataFields.VALUE
            ]
        )
        height: Meter = Meter(
            clinical_demo_file[ClinicalDemographicDataFields.HEIGHT][
                ClinicalDemographicDataFields.VALUE
            ]
        )
        return ClinicalDemographicData(name, age, sex, weight, height)
