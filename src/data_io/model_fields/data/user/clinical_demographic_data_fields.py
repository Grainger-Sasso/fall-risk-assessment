from enum import Enum


class ClinicalDemographicDataFields(Enum):
    """Fields used in clinical demographic data data files"""

    DEMOGRAPHIC_DATA = "demographic_data"
    NAME = "name"
    AGE = "age"
    SEX = "sex"
    WEIGHT = "weight"
    HEIGHT = "height"
    IDENTIFIER = "identifier"
    FALLER_STATUS = "faller_status"
    VALUE = "value"
    UNIT = "unit"
