import pandas as pd # type: ignore
from pathlib import Path
from typing import Tuple

from src.data_io.formats.csv.csv_file import CSVFile
from src.data_io.read_write.writers.file_writer import FileWriter


class CSVFileWriter(FileWriter):
    """
    CSV file writer.
    """

    def write(self, path: Path, csv_file: CSVFile, **kwargs) -> Tuple[bool, str]:
        """
        Writes CSV data to a file.

        Args:
            path (Path): File path to write.
            data (CSVFile): Data to write.
            **kwargs: Additional parameters for specific file formats (e.g., compression, delimiter).

        Returns:
            Tuple[bool, str]: (success flag, error message)
        """
        try:
            df = pd.DataFrame(csv_file.data)
            df.to_csv(path, index=False)
            return True, ""
        except Exception as e:
            return False, str(e)
