import tempfile
from pathlib import Path

from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.import_export.importers.data.user.user_data_importer import (
    UserDataFileNames,
    UserDataImporter,
)
from src.data_model.data.user.clinical.clinical_demographic_data import (
    ClinicalDemographicData,
)
from src.data_model.data.user.user_data import UserData
from src.identifiers.user.clinical_identifier import ClinicalIdentifier
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import TestConstants, UserDataHelper


class TestUserDataImporter(BaseTest):
    def setUp(self):
        self.importer = UserDataImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.helper = UserDataHelper()

        # Create test files in temp directory
        self.user_data_path = self.helper.create_test_user_data_file(
            self.temp_path / f"{UserDataFileNames.USER_DATA.value}.json"
        )
        self.clinical_data_path = self.helper.create_test_clinical_demographic_file(
            self.temp_path
            / f"{UserDataFileNames.CLININCAL_DEMOGRAPHIC_DATA.value}.json"
        )

    def tearDown(self):
        # Clean up test files
        if self.user_data_path.exists():
            self.user_data_path.unlink()
        if self.clinical_data_path.exists():
            self.clinical_data_path.unlink()
        self.temp_path.rmdir()

    def test_import_data(self):
        # Import data from test directory
        result = self.importer.import_data(self.temp_path)

        # Verify result type
        self.assertIsInstance(result, UserData)

        # Verify user data
        self.assertEqual(result.user_identifier.value, TestConstants.USER_DATA_ID.value)

        # Verify clinical data
        self.assertIsInstance(result.clinical_demographic_data, ClinicalDemographicData)
        self.assertEqual(
            result.clinical_demographic_data.name, TestConstants.USER_NAME.value
        )
        self.assertEqual(
            result.clinical_demographic_data.age, TestConstants.USER_AGE.value
        )
        self.assertEqual(
            result.clinical_demographic_data.sex.value, TestConstants.USER_SEX.value
        )
        self.assertEqual(
            result.clinical_demographic_data.height.value,
            TestConstants.USER_HEIGHT.value,
        )
        self.assertEqual(
            result.clinical_demographic_data.weight.value,
            TestConstants.USER_WEIGHT.value,
        )
        self.assertIsInstance(
            result.clinical_demographic_data.clinical_identifier, ClinicalIdentifier
        )
        self.assertEqual(
            result.clinical_demographic_data.clinical_identifier.value,
            TestConstants.CLINICAL_ID.value,
        )

    def test_missing_file(self):
        # Remove one of the required files
        self.user_data_path.unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)


if __name__ == "__main__":
    TestUserDataImporter.run_tests()
