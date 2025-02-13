import csv
from pathlib import Path

from src.data_io.read_write.readers.file_reader import FileReader
from src.data_io.formats.csv.csv_file import CSVFile


class CSVFileReader(FileReader):
    """
    CSV file reader
    """

    def read(self, path: Path) -> CSVFile:
        """
        Reads a CSV file and creates a CSVFile object.

        Args:
            path (Path): The path to the CSV file.

        Returns:
            CSVFile: An instance of CSVFile containing the column headers and data.
        """
        with path.open(mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames  # Get column headers
            data = {header: [] for header in fieldnames}  # Initialize data dictionary

            for row in reader:
                for header in fieldnames:
                    data[header].append(row[header])

        return CSVFile(fieldnames=fieldnames, data=data)
