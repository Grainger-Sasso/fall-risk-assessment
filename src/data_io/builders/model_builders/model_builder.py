from abc import ABC, abstractmethod
from typing import Any

from src.data_io.formats.file_format import FileFormat


class ModelBuilder(ABC):
    """Abstract base class for building model objects from file formats.

    This class defines the interface for all model builders, which convert
    file format objects into domain model objects.

    Attributes:
        version (str): Version identifier for the builder implementation
    """

    version: str

    def __init_subclass__(cls) -> None:
        """Validate subclass implementation.

        Raises:
            TypeError: If version is not defined in subclass
        """
        super().__init_subclass__()
        if not hasattr(cls, "version"):
            raise TypeError(f"{cls.__name__} must define class variable 'version'")

    def __init__(self):
        """Initialize the model builder.

        The version is copied from the class variable to instance variable
        to allow potential runtime version changes.
        """
        self.version = self.__class__.version

    @abstractmethod
    def build(self, input_file: FileFormat, **kwargs) -> Any:
        """Converts input file to data model

        Args:
            input_file (FileFormat): Input file data to convert
            **kwargs: Additional build parameters

        Returns:
            Any: Constructed model object

        Raises:
            ValueError: If input file data is invalid
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Subclass must implement build method")

    def _validate_input_file(self, input_file: FileFormat) -> None:
        """Validate input file format.

        Args:
            input_file (FileFormat): Input file to validate

        Raises:
            ValueError: If input file is None or invalid
        """
        if not input_file:
            raise ValueError("Input file data is required")
        if not isinstance(input_file, FileFormat):
            raise ValueError(f"Expected FileFormat, got {type(input_file)}")
