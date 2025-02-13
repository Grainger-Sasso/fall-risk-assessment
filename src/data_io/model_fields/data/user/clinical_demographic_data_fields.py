from enum import Enum


class ClinicalDemographicDataFields(Enum):
    """Fields used in clinical demographic data data files"""

    DEMOGRAPHIC_DATA = "demographic_data"
    NAME = "name"
    AGE = "age"
    SEX = "sex"
    WEIGHT = "weight"
    HEIGHT = "height"
    VALUE = "value"
    UNIT = "unit"
