from src.data_io.builders.model_builders.model_builder import ModelBuilder
from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.model_fields.data.user.clinical_demographic_data_fields import (
    ClinicalDemographicDataFields,
)
from src.data_io.model_fields.data.user.user_data_fields import UserDataFields
from src.data_model.data.user.clinical.clinical_assessment import ClinicalAssessment
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
    Sex,
)
from src.data_model.data.user.user_data import UserData
from src.identifiers.user.clinical_identifier import ClinicalIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.mechanics.units.si.kilogram import Kilogram
from src.util.mechanics.units.si.meter import Meter


class UserDataBuilder(ModelBuilder):
    """User data builder"""

    version = "1.0"

    def build(
        self, input_file: JSONDictFile, clinical_demo_file: JSONDictFile
    ) -> UserData:
        if not isinstance(input_file, JSONDictFile) or not isinstance(
            clinical_demo_file, JSONDictFile
        ):
            raise ValueError("Invalid User data JSON file")
        return self.__build_user_data(input_file, clinical_demo_file)

    def __build_user_data(
        self, input_file: JSONDictFile, clinical_demo_file: JSONDictFile
    ) -> UserData:
        user_identifier: UserIdentifier = UserIdentifier(
            input_file.data[UserDataFields.USER_DATA_IDENTIFIER.value]
        )
        clinical_demo_data: ClinicalDemographicData = (
            self.__build_clinical_demographic_data(clinical_demo_file)
        )
        clinical_fall_risk_assessment: ClinicalAssessment = ClinicalAssessment()
        return UserData(
            user_identifier, clinical_demo_data, [clinical_fall_risk_assessment]
        )

    def __build_clinical_demographic_data(
        self, clinical_demo_file: JSONDictFile
    ) -> ClinicalDemographicData:
        name: str = clinical_demo_file.data[ClinicalDemographicDataFields.NAME.value][
            ClinicalDemographicDataFields.VALUE.value
        ]
        age: float = clinical_demo_file.data[ClinicalDemographicDataFields.AGE.value][
            ClinicalDemographicDataFields.VALUE.value
        ]
        sex: Sex = Sex(
            clinical_demo_file.data[ClinicalDemographicDataFields.SEX.value][
                ClinicalDemographicDataFields.VALUE.value
            ].lower()
        )
        weight: Kilogram = Kilogram(
            clinical_demo_file.data[ClinicalDemographicDataFields.WEIGHT.value][
                ClinicalDemographicDataFields.VALUE.value
            ]
        )
        height: Meter = Meter(
            clinical_demo_file.data[ClinicalDemographicDataFields.HEIGHT.value][
                ClinicalDemographicDataFields.VALUE.value
            ]
        )
        clinical_identifier: ClinicalIdentifier = ClinicalIdentifier(
            clinical_demo_file.data[ClinicalDemographicDataFields.IDENTIFIER.value][
                ClinicalDemographicDataFields.VALUE.value
            ]
        )
        return ClinicalDemographicData(
            name, age, sex, weight, height, clinical_identifier
        )
