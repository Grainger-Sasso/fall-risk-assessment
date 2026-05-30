from typing import Type

from src.identifiers.feature.aggregate_feature_identifier import AggregateFeatureIdentifier
from src.identifiers.feature.feature_identifier import FeatureIdentifier
from src.identifiers.feature.raw_feature_identifier import RawFeatureIdentifier
from src.identifiers.identifier import Identifier
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument_specification.instrument_specification_identifier import (
    InstrumentSpecificationIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier


class IdentifierTypeRegistry:
    """Maps Identifier classes to normalized sqlite type names."""

    TYPE_TO_NAME = {
        UserIdentifier: "user_data",
        IMUDataIdentifier: "imu_data",
        FeatureIdentifier: "feature",
        RawFeatureIdentifier: "raw_feature",
        AggregateFeatureIdentifier: "aggregate_feature",
        InstrumentSpecificationIdentifier: "instrument_specification",
    }
    NAME_TO_TYPE = {v: k for k, v in TYPE_TO_NAME.items()}

    DEFAULT_RELATIONS = {
        ("imu_data", "user_data"): "imu_to_user",
        ("feature", "imu_data"): "feature_to_imu",
        ("raw_feature", "imu_data"): "raw_to_imu",
        ("aggregate_feature", "raw_feature"): "agg_to_raw",
    }

    @classmethod
    def get_type_name(cls, id_type: Type[Identifier]) -> str:
        if id_type in cls.TYPE_TO_NAME:
            return cls.TYPE_TO_NAME[id_type]
        return id_type.__name__.replace("Identifier", "").lower()

    @classmethod
    def build_identifier(cls, id_type_name: str, value: str) -> Identifier:
        if id_type_name not in cls.NAME_TO_TYPE:
            raise KeyError(f"Unknown id type name: {id_type_name}")
        return cls.NAME_TO_TYPE[id_type_name](value)

    @classmethod
    def infer_relation_type(cls, source_type_name: str, target_type_name: str) -> str:
        return cls.DEFAULT_RELATIONS.get(
            (source_type_name, target_type_name),
            f"{source_type_name}_to_{target_type_name}",
        )
