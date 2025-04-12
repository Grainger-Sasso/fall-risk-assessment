from abc import ABC, abstractmethod

from src.identifiers.identifier import Identifier


class AssessmentData(ABC):

    @abstractmethod
    def get_data_id(self) -> Identifier:
        pass
