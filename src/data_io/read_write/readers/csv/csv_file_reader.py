import csv
from pathlib import Path

from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.read_write.readers.file_reader import FileReader


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
            original_fieldnames = reader.fieldnames  # Get column headers

            # Clean BOM from fieldnames if present
            fieldnames = self._clean_bom_from_fieldnames(original_fieldnames)

            # Create mapping from original fieldnames to cleaned fieldnames
            fieldname_map = {
                old: new for old, new in zip(original_fieldnames, fieldnames)
            }

            # Initialize data dictionary with cleaned fieldnames
            data = {header: [] for header in fieldnames}

            for row in reader:
                for original_header, cleaned_header in fieldname_map.items():
                    data[cleaned_header].append(row[original_header])

        return CSVFile(fieldnames=fieldnames, data=data)

    def _clean_bom_from_fieldnames(self, fieldnames):
        """
        Removes BOM from fieldnames if present.

        Args:
            fieldnames (list): List of fieldnames that might contain BOM

        Returns:
            list: List of fieldnames with BOM removed
        """
        bom = "\ufeff"
        has_bom = any(fieldname.startswith(bom) for fieldname in fieldnames)

        if has_bom:
            return [fieldname.replace(bom, "") for fieldname in fieldnames]

        return fieldnames
