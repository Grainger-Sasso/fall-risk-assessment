from src.data_io.builders.model_builders.data.user.user_data_builder import (
    UserDataBuilder,
)
from src.data_model.data.user.clinical.clinical_assessment import ClinicalAssessment
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
    Sex,
)
from src.data_model.data.user.user_data import UserData
from src.identifiers.user.user_identifier import UserIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import TestConstants, UserDataHelper


class TestUserDataBuilder(BaseTest):
    def setUp(self):
        self.builder = UserDataBuilder()
        self.data_helper = UserDataHelper()

    def test_build_valid_data(self):
        # Create test JSON data
        user_json = self.data_helper.create_test_user_data_json()
        clinical_demo_json = self.data_helper.create_test_clinical_demographic_json()

        # Test building user data
        result = self.builder.build(user_json, clinical_demo_json)

        # Assertions
        self.assertIsInstance(result, UserData)
        self.assertIsInstance(result.user_identifier, UserIdentifier)
        self.assertEqual(result.user_identifier.value, TestConstants.USER_DATA_ID.value)

        # Test clinical demographic data
        demo_data = result.clinical_demographic_data
        self.assertIsInstance(demo_data, ClinicalDemographicData)
        self.assertEqual(demo_data.name, TestConstants.USER_NAME.value)
        self.assertEqual(demo_data.age, TestConstants.USER_AGE.value)
        self.assertEqual(demo_data.sex, Sex(TestConstants.USER_SEX.value.lower()))
        self.assertEqual(demo_data.weight.value, TestConstants.USER_WEIGHT.value)
        self.assertEqual(demo_data.height.value, TestConstants.USER_HEIGHT.value)

        # Test clinical assessment
        self.assertIsInstance(result.clinical_assessments, list)
        self.assertEqual(len(result.clinical_assessments), 1)
        self.assertIsInstance(result.clinical_assessments[0], ClinicalAssessment)

    def test_build_empty_data(self):
        with self.assertRaises(ValueError):
            self.builder.build(None, None)


if __name__ == "__main__":
    TestUserDataBuilder.run_tests()
