import tempfile
from pathlib import Path

from src.data_io.formats.json.json_dict_file import JSONDictFile
from src.data_io.read_write.readers.json.json_dict_file_reader import JSONDictFileReader
from src.data_io.read_write.writers.json.json_dict_file_writer import JSONDictFileWriter
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FileIOHelper


class TestJSONFileIO(BaseTest):
    def setUp(self):
        self.reader = JSONDictFileReader()
        self.writer = JSONDictFileWriter()
        self.helper = FileIOHelper()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir) / "test.json"

    def tearDown(self):
        if self.temp_path.exists():
            self.temp_path.unlink()
        self.temp_path.parent.rmdir()

    def test_write_and_read_json(self):
        # Create test data
        test_data: JSONDictFile = self.helper.create_test_json_file()

        # Write data
        success, error = self.writer.write(self.temp_path, test_data)
        self.assertTrue(success, f"Failed to write JSON: {error}")
        self.assertTrue(self.temp_path.exists())

        # Read data back
        result: JSONDictFile = self.reader.read(self.temp_path)

        # Verify data
        self.assertIsInstance(result, JSONDictFile)
        self.assertEqual(result.data, test_data.data)

    def test_read_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.reader.read(Path("nonexistent.json"))

    def test_read_invalid_extension(self):
        invalid_path = Path(self.temp_dir) / "test.txt"
        with self.assertRaises(ValueError):
            self.reader.read(invalid_path)


if __name__ == "__main__":
    TestJSONFileIO.run_tests() 