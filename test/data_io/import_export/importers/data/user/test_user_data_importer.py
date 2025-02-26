import json
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
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import TestConstants


class TestUserDataImporter(BaseTest):
    def setUp(self):
        self.importer = UserDataImporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test JSON files
        self.user_data = {
            "user_data_identifier": TestConstants.USER_DATA_ID.value,
        }

        self.clinical_data = {
            "name": {"value": TestConstants.USER_NAME.value, "unit": None},
            "age": {"value": TestConstants.USER_AGE.value, "unit": "years"},
            "sex": {"value": TestConstants.USER_SEX.value, "unit": None},
            "weight": {"value": TestConstants.USER_WEIGHT.value, "unit": "kilograms"},
            "height": {"value": TestConstants.USER_HEIGHT.value, "unit": "centimeters"},
        }

        # Write test files
        self._write_json_file(
            self.temp_path / f"{UserDataFileNames.USER_DATA.value}.json", self.user_data
        )
        self._write_json_file(
            self.temp_path
            / f"{UserDataFileNames.CLININCAL_DEMOGRAPHIC_DATA.value}.json",
            self.clinical_data,
        )

    def tearDown(self):
        # Clean up test files
        for file in self.temp_path.glob("*.json"):
            file.unlink()
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

    def test_missing_file(self):
        # Remove one of the required files
        (self.temp_path / f"{UserDataFileNames.USER_DATA.value}.json").unlink()

        # Verify import raises error
        with self.assertRaises(FileNotFoundError):
            self.importer.import_data(self.temp_path)

    def _write_json_file(self, path: Path, data: dict) -> None:
        """Helper method to write test JSON files."""
        with open(path, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    TestUserDataImporter.run_tests()
