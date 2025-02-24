import tempfile
from pathlib import Path

from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.read_write.readers.csv.csv_file_reader import CSVFileReader
from src.data_io.read_write.writers.csv.csv_file_writer import CSVFileWriter
from test.base_test import BaseTest
from test.data_io.test_data.test_data_helper import FileIOHelper


class TestCSVFileIO(BaseTest):
    def setUp(self):
        self.reader = CSVFileReader()
        self.writer = CSVFileWriter()
        self.helper = FileIOHelper()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir) / "test.csv"

    def tearDown(self):
        if self.temp_path.exists():
            self.temp_path.unlink()
        self.temp_path.parent.rmdir()

    def test_write_and_read_csv(self):
        # Create test data
        test_data: CSVFile = self.helper.create_test_csv_file()

        # Write data
        success, error = self.writer.write(self.temp_path, test_data)
        self.assertTrue(success, f"Failed to write CSV: {error}")
        self.assertTrue(self.temp_path.exists())

        # Read data back
        result: CSVFile = self.reader.read(self.temp_path)

        # Verify data
        self.assertIsInstance(result, CSVFile)
        self.assertEqual(result.fieldnames, test_data.fieldnames)
        self.assertEqual(len(result.data), len(test_data.data))
        for field in test_data.fieldnames:
            self.assertEqual(result.data[field], test_data.data[field])

    def test_read_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.reader.read(Path("nonexistent.csv"))


if __name__ == "__main__":
    TestCSVFileIO.run_tests()
